import sys, os, math, warnings
import numpy as np
import scipy.special as sp
from scipy import sparse
from scipy.sparse.linalg import spsolve

from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QGridLayout, QPushButton, QLabel, QLineEdit, QRadioButton,
    QButtonGroup, QListWidget, QFileDialog, QMessageBox, QStatusBar,
    QTableWidget, QTableWidgetItem, QHeaderView
)
import pyqtgraph as pg

# ==============================================================================
# BrPLS Baseline Estimation Algorithm from nonparams_est.py
# ==============================================================================
class NONPARAMS_EST:
    def __init__(self, y):
        """
        y: 1D numpy array (e.g., np.log(selected_frame))
        """
        self.data = y

    def pls(self, method='BrPLS', l=1e6, ratio=1e-6, nitermax=50):
        '''
        Bayesian Asymmetrically reweighted penalized least squares (BrPLS)
        @ Q. Wang. Phys. Rev. E. (2022)
        '''
        L, beta = len(self.data), 0.5
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
        D = l * D.dot(D.transpose())
        w, z = np.ones(L), self.data.copy()
        warnings.filterwarnings("ignore")
        for i in range(nitermax):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + D
            zt = spsolve(Z, w * self.data)
            d = self.data - zt
            
            # Avoid division by zero
            pos_mask = d > 0
            neg_mask = d < 0
            d_m = np.mean(d[pos_mask]) if np.any(pos_mask) else 1e-6
            d_sigma = np.sqrt(np.mean(d[neg_mask]**2)) if np.any(neg_mask) else 1e-6
            if d_m == 0: d_m = 1e-6
            if d_sigma == 0: d_sigma = 1e-6
            
            w = 1 / (1 + beta / (1 - beta) * np.sqrt(np.pi / 2) * d_sigma / d_m * (1 + sp.erf((d / d_sigma - d_sigma / d_m) / np.sqrt(2))) * np.exp((d / d_sigma - d_sigma / d_m)**2 / 2))
            
            if np.sqrt(np.sum((z - zt)**2) / np.sum(z**2)) < ratio: 
                break
            z = zt
            if np.abs(beta + np.mean(w) - 1.) < ratio: 
                break
            beta = 1 - np.mean(w)
        return z

# ==============================================================================
# Synchronized ViewBox for 1D/2D Plot Linking
# ==============================================================================
class LinkedViewBox(pg.ViewBox):
    def __init__(self, master_view=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.master_view = master_view

    def set_master(self, master_view):
        self.master_view = master_view

    def mouseDragEvent(self, ev, axis=None):
        super().mouseDragEvent(ev, axis)
        if self.master_view:
            self.master_view.sync_views(self)

    def wheelEvent(self, ev, axis=None):
        super().wheelEvent(ev, axis)
        if self.master_view:
            self.master_view.sync_views(self)


# ==============================================================================
# Main Monitor Window
# ==============================================================================
class DataMonitorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RF Data Monitor, Baseline Estimator & Peak Finder")
        self.resize(1650, 950)
        
        # State Data
        self.freqs = None
        self.times = None
        self.psd_arrays = None  # Shape: [NumFrames, NumFreqs]
        
        self.global_baseline = None       # Computed log-baseline
        self.baseline_source_frame = None # Index of the frame used for calculation
        self.current_frame_idx = 0        # Linked with the red dragline
        
        self.updating_line = False
        self.current_dir = os.getcwd()
        
        # UI peak region markers
        self.peak_region_plots = []
        
        # NEW: 用于在内部保存已经点选过的帧的数据和帧号追踪
        self.saved_csv_data = []         # 结构: [ [frame_idx, freq_center, area, mean, sigma], ... ]
        self.saved_frames_set = set()     # 用于统计不重复的写了多少个不同的 frames
        
        self.init_ui()

    def init_ui(self):
        # Main central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 3-Column Horizontal layout
        main_layout = QHBoxLayout(central_widget)
        
        # ----------------------------------------------------------------------
        # COLUMN 1: Waterfall Display (Left)
        # ----------------------------------------------------------------------
        col1_layout = QVBoxLayout()
        col1_label = QLabel("Waterfall View (Time-Frequency)")
        col1_label.setStyleSheet("font-weight: bold; font-size: 13px; color: #333;")
        col1_layout.addWidget(col1_label)

        # Using GraphicsLayoutWidget to properly lay out PlotItem & ColorBarItem side-by-side
        self.waterfall_canvas = pg.GraphicsLayoutWidget()
        col1_layout.addWidget(self.waterfall_canvas)
        
        # Add PlotItem to Canvas
        self.waterfall_plot = self.waterfall_canvas.addPlot(row=0, col=0)
        self.waterfall_plot.setLabel('left', 'Time (Frame Index)')
        self.waterfall_plot.setLabel('bottom', 'Frequency (Hz)')
        
        # Image item for rendering 2D PSD array
        self.img_item = pg.ImageItem()
        self.waterfall_plot.addItem(self.img_item)

        # Horizontal infinite line for frame selection (movable)
        self.inf_line = pg.InfiniteLine(pos=0, angle=0, movable=True, pen=pg.mkPen('r', width=2))
        self.inf_line.sigDragged.connect(self.on_line_dragged)
        self.waterfall_plot.addItem(self.inf_line)

        # Built-in PyQtGraph ColorBar (Placed in adjacent grid column to prevent collision)
        self.colorbar = pg.ColorBarItem(colorMap='viridis', label='Power Spectral Density [arb. unit]', rounding=0.0000001, limits=(None,None))
        self.colorbar.setImageItem(self.img_item)
        self.waterfall_canvas.addItem(self.colorbar, row=0, col=1)

        main_layout.addLayout(col1_layout, stretch=4)

        # ----------------------------------------------------------------------
        # COLUMN 2: Cut slices (Middle)
        # ----------------------------------------------------------------------
        col2_layout = QVBoxLayout()
        
        # Top 1D Plot: Raw data (Blue) + Baseline (Orange)
        col2_label_top = QLabel("Selected Frame Spectrum & Estimated Baseline")
        col2_label_top.setStyleSheet("font-weight: bold; font-size: 13px; color: #333;")
        col2_layout.addWidget(col2_label_top)
        
        self.vb_top = LinkedViewBox(master_view=self)
        self.top_plot = pg.PlotWidget(viewBox=self.vb_top, title="Raw Data (Blue) vs Baseline (Orange)")
        self.top_plot.setLabel('left', 'Intensity')
        self.top_plot.setLabel('bottom', 'Frequency (Hz)')
        
        self.curve_raw = self.top_plot.plot(pen=pg.mkPen('#1f77b4', width=1.5), name="Raw")
        self.curve_base = self.top_plot.plot(pen=pg.mkPen('#ff7f0e', width=2.0), name="Baseline")
        col2_layout.addWidget(self.top_plot)
        
        # Bottom 1D Plot: Subtracted spectrum
        col2_label_bottom = QLabel("Baseline Subtraction Output")
        col2_label_bottom.setStyleSheet("font-weight: bold; font-size: 13px; color: #333;")
        col2_layout.addWidget(col2_label_bottom)
        
        self.vb_bottom = LinkedViewBox(master_view=self)
        self.bottom_plot = pg.PlotWidget(viewBox=self.vb_bottom, title="Result (PSD - Baseline)")
        self.bottom_plot.setLabel('left', 'Subtraction')
        self.bottom_plot.setLabel('bottom', 'Frequency (Hz)')
        
        self.curve_sub = self.bottom_plot.plot(pen=pg.mkPen('#2ca02c', width=1.5))
        
        # NEW: Horizontal Threshold Line for Peak Finding
        self.threshold_line = pg.InfiniteLine(pos=0.5, angle=0, movable=True, pen=pg.mkPen('m', width=1.5, style=Qt.DashLine))
        self.bottom_plot.addItem(self.threshold_line)
        
        col2_layout.addWidget(self.bottom_plot)
        
        main_layout.addLayout(col2_layout, stretch=4)

        # Connect linked view boxes for mouse interactions
        self.vb_wf = self.waterfall_plot.vb
        self.linked_vbs = [self.vb_wf, self.vb_top, self.vb_bottom]
        self.vb_top.set_master(self)
        self.vb_bottom.set_master(self)
        self.waterfall_plot.sigRangeChanged.connect(self.on_waterfall_range_changed)

        # ----------------------------------------------------------------------
        # COLUMN 3: Control Panel & Results (Right)
        # ----------------------------------------------------------------------
        col3_layout = QVBoxLayout()
        col3_layout.setSpacing(10)
        
        col3_label = QLabel("Control Options")
        col3_label.setStyleSheet("font-weight: bold; font-size: 13px; color: #333;")
        col3_layout.addWidget(col3_label)
        
        # 1. Directory Selection & File List
        dir_btn = QPushButton("Select Directory")
        dir_btn.clicked.connect(self.on_select_directory)
        col3_layout.addWidget(dir_btn)
        
        self.file_list_widget = QListWidget()
        self.file_list_widget.itemDoubleClicked.connect(self.on_file_double_clicked)
        self.file_list_widget.setFixedHeight(120)
        col3_layout.addWidget(self.file_list_widget)
        
        # 2. Log / Linear Scale Selection
        scale_group_box = QWidget()
        scale_grid = QHBoxLayout(scale_group_box)
        scale_grid.setContentsMargins(0, 0, 0, 0)
        
        self.radio_log = QRadioButton("Log Scale")
        self.radio_linear = QRadioButton("Linear Scale")
        self.radio_log.setChecked(True)  # default
        
        self.radio_group = QButtonGroup()
        self.radio_group.addButton(self.radio_log)
        self.radio_group.addButton(self.radio_linear)
        self.radio_group.buttonClicked.connect(self.on_scale_changed)
        
        scale_grid.addWidget(self.radio_log)
        scale_grid.addWidget(self.radio_linear)
        col3_layout.addWidget(scale_group_box)
        
        # 3. Parameters Input Block
        param_widget = QWidget()
        param_grid = QGridLayout(param_widget)
        param_grid.setContentsMargins(0, 0, 0, 0)
        
        param_grid.addWidget(QLabel("Est. Frame Index:"), 0, 0)
        self.baseline_frame_input = QLineEdit("0")
        param_grid.addWidget(self.baseline_frame_input, 0, 1)
        
        param_grid.addWidget(QLabel("Smoothness (l):"), 1, 0)
        self.smoothness_l_input = QLineEdit("1e6")
        param_grid.addWidget(self.smoothness_l_input, 1, 1)
        
        col3_layout.addWidget(param_widget)
        
        # 4. Action Button for Baseline
        apply_btn = QPushButton("Apply Parameters")
        apply_btn.setStyleSheet("background-color: #0078d4; color: white; font-weight: bold; height: 30px;")
        apply_btn.clicked.connect(self.on_apply_parameters)
        col3_layout.addWidget(apply_btn)
        
        # 5. Peak Finding Section
        col3_layout.addWidget(QLabel("Peak Extraction"))
        
        peak_btn_layout = QHBoxLayout()
        self.lbl_peak_frame = QLabel("#Frame ")
        
        self.peak_btn = QPushButton("Find Peaks")
        self.peak_btn.setStyleSheet("background-color: #107c41; color: white; font-weight: bold; height: 30px;")
        self.peak_btn.clicked.connect(self.on_find_peaks)
        
        peak_btn_layout.addWidget(self.lbl_peak_frame, stretch=3)
        peak_btn_layout.addWidget(self.peak_btn, stretch=2)
        col3_layout.addLayout(peak_btn_layout)

        # 6. Results Table Widget（修改：由于保存需要区分Frame，给UI可视化增加一列#Frame展现更加直观）
        self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(5)
        self.table_widget.setHorizontalHeaderLabels(["#Frame", "Freq Center", "Area", "Mean (1st)", "Sigma (2nd)"])
        self.table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_widget.verticalHeader().setVisible(False)
        self.table_widget.setEditTriggers(QTableWidget.NoEditTriggers)
        col3_layout.addWidget(self.table_widget)

        # 7. File Save (修复：将其加入到了右侧布局面板的最底部)
        file_save_layout = QHBoxLayout()
        self.lbl_file_count = QLabel("Writted 0 Frames")
        self.clear_btn = QPushButton("Clear .csv")
        self.save_btn = QPushButton("Save .csv")
        
        # 绑定按钮点击事件槽函数
        self.clear_btn.clicked.connect(self.on_clear_csv)
        self.save_btn.clicked.connect(self.on_save_csv)
        
        file_save_layout.addWidget(self.lbl_file_count)
        file_save_layout.addWidget(self.clear_btn)
        file_save_layout.addWidget(self.save_btn)
        col3_layout.addLayout(file_save_layout)
        
        main_layout.addLayout(col3_layout, stretch=3)
        
        # Status Bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready. Please select a directory to begin.")
        
        # Track mouse hover
        self.waterfall_plot.scene().sigMouseMoved.connect(self.on_waterfall_mouse_moved)
        self.top_plot.scene().sigMouseMoved.connect(self.on_top_mouse_moved)
        self.bottom_plot.scene().sigMouseMoved.connect(self.on_bottom_mouse_moved)

        # Initial population of current working directory
        self.populate_file_list()

    # ==============================================================================
    # Directory & File Handling
    # ==============================================================================
    def populate_file_list(self):
        self.file_list_widget.clear()
        if not os.path.exists(self.current_dir):
            return
        files = [f for f in os.listdir(self.current_dir) if f.endswith('.npz')]
        self.file_list_widget.addItems(files)

    def on_file_double_clicked(self, item):
        filename = item.text()
        filepath = os.path.join(self.current_dir, filename)
        
        try:
            data = np.load(filepath)
            self.freqs = data['frequencies']
            self.times = data['times']
            self.psd_arrays = data['psd_arrays']

            num_frames, num_freqs = self.psd_arrays.shape
            
            # Setup image scaling bounds:
            freq_min, freq_max = self.freqs[0], self.freqs[-1]
            time_min, time_max = 0.0, float(num_frames - 1)
            
            x_range = freq_max - freq_min
            y_range = time_max - time_min

            # Clear old plots, baseline state and peak table
            self.global_baseline = None
            self.baseline_source_frame = None
            self.lbl_peak_frame.setText("#Frame ")
            self.table_widget.setRowCount(0)
            self.clear_peak_markers()
            
            # 同时在新导入文件时，自动清空上一个文件的内部临时CSV数据
            self.saved_csv_data.clear()
            self.saved_frames_set.clear()
            self.lbl_file_count.setText("Writted 0 Frames")
            
            # Populate image item
            self.img_item.setImage(self.psd_arrays.T)
            self.img_item.setRect(QRectF(freq_min, time_min, x_range, y_range))
            
            # Enforce limits to avoid drifting
            self.waterfall_plot.setLimits(xMin=freq_min, xMax=freq_max, yMin=time_min, yMax=time_max)
            self.top_plot.setLimits(xMin=freq_min, xMax=freq_max)
            self.bottom_plot.setLimits(xMin=freq_min, xMax=freq_max)

            # Auto fit view
            self.waterfall_plot.autoRange()
            self.top_plot.autoRange()
            self.bottom_plot.autoRange()

            mid_frame = num_frames // 2
            self.baseline_frame_input.setText(str(mid_frame))
            self.smoothness_l_input.setText("1e6")
            
            # Position Drag-Line to Mid Frame
            self.current_frame_idx = mid_frame
            self.updating_line = True
            self.inf_line.setValue(mid_frame)
            self.updating_line = False
            
            # Update auto contrast levels
            self.colorbar.setLevels((self.psd_arrays.min(), self.psd_arrays.max()))
            
            # Set default threshold line position based on general range
            self.threshold_line.setValue(0.0)
            
            # Initial drawing
            self.update_slice_plots()
            self.statusBar.showMessage(f"Successfully loaded: {filename}", 4000)
            
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load .npz contents:\n{str(e)}")

    # ==============================================================================
    # View Synchronization & Navigation
    # ==============================================================================
    def sync_views(self, sender_vb):
        """Broadcasts X-axis changes from one view to all linked views."""
        if self.freqs is None:
            return
        
        sender_range = sender_vb.viewRange()
        x_min, x_max = sender_range[0]
        
        for vb in self.linked_vbs:
            if vb != sender_vb:
                vb.blockSignals(True)
                vb.setXRange(x_min, x_max, padding=0)
                vb.blockSignals(False)

    def on_waterfall_range_changed(self):
        self.sync_views(self.vb_wf)

    # ==============================================================================
    # Slit Positioning & User Parameters Application
    # ==============================================================================
    def on_line_dragged(self, line):
        if self.psd_arrays is None or self.updating_line:
            return
        
        num_frames = self.psd_arrays.shape[0]
        val = line.value()
        
        clamped = max(0, min(num_frames - 1, int(round(val))))
        self.current_frame_idx = clamped
        
        self.updating_line = True
        self.inf_line.setValue(clamped)
        self.updating_line = False
        
        self.update_slice_plots()

    def on_apply_parameters(self):
        if self.psd_arrays is None:
            QMessageBox.warning(self, "Missing Data", "Please load a .npz file first.")
            return
        
        try:
            frame_idx = int(self.baseline_frame_input.text().strip())
            smoothness_l = float(self.smoothness_l_input.text().strip())
        except ValueError:
            QMessageBox.warning(self, "Invalid Inputs", "Please verify Frame Index is integer and Smoothness is numeric.")
            return

        num_frames = self.psd_arrays.shape[0]
        if frame_idx < 0 or frame_idx >= num_frames:
            QMessageBox.warning(self, "Out of Range", f"Frame index must be between 0 and {num_frames - 1}.")
            return
        
        self.statusBar.showMessage("Computing BrPLS baseline...")
        QApplication.processEvents()
        
        try:
            selected_frame = self.psd_arrays[frame_idx, :]
            
            # Perform log estimation
            floor_safe = np.where(selected_frame <= 0, 1e-16, selected_frame)
            log_frame = np.log(floor_safe)
            
            estimator = NONPARAMS_EST(log_frame)
            log_baseline = estimator.pls(method='BrPLS', l=smoothness_l)
            
            self.global_baseline = np.exp(log_baseline)
            self.baseline_source_frame = frame_idx
            
            # Re-render plots and clean old markers
            self.clear_peak_markers()
            self.table_widget.setRowCount(0)
            self.update_slice_plots()
            self.statusBar.showMessage("Baseline calculation complete.", 4000)
            
        except Exception as e:
            QMessageBox.critical(self, "Calculation Failed", f"An error occurred during BrPLS calculation:\n{str(e)}")

    def on_scale_changed(self, btn):
        self.update_slice_plots()

    # ==============================================================================
    # Peak Finding Algorithm Implementation & CSV Operations
    # ==============================================================================
    def clear_peak_markers(self):
        for item in self.peak_region_plots:
            self.bottom_plot.removeItem(item)
        self.peak_region_plots.clear()

    def on_find_peaks(self):
        if self.psd_arrays is None or self.global_baseline is None or self.baseline_source_frame is None:
            QMessageBox.warning(self, "Baseline Required", "Please calculate the baseline via 'Apply Parameters' first!")
            return

        raw_slice = self.psd_arrays[self.current_frame_idx, :]
        linear_subtracted = raw_slice - self.global_baseline
        num_points = len(linear_subtracted)

        threshold_visual = self.threshold_line.value()
        
        if self.radio_log.isChecked():
            linear_threshold = 10 ** threshold_visual
        else:
            linear_threshold = threshold_visual

        above_threshold = linear_subtracted > linear_threshold
        
        regions = []
        in_region = False
        start_idx = -1
        
        for i in range(num_points):
            if above_threshold[i]:
                if not in_region:
                    start_idx = i
                    in_region = True
            else:
                if in_region:
                    regions.append((start_idx, i - 1))
                    in_region = False
        if in_region:
            regions.append((start_idx, num_points - 1))

        if not regions:
            self.clear_peak_markers()
            self.table_widget.setRowCount(0)
            self.statusBar.showMessage("No peaks detected exceeding the threshold.", 4000)
            return

        final_peaks_info = []
        self.clear_peak_markers()

        # Step 2: Expand each detected candidate region to true zero-crossing limits
        expanded_bounds = []
        for start, end in regions:
            left_bound = start
            while left_bound > 0 and linear_subtracted[left_bound] >= 0:
                left_bound -= 1
            left_bound = max(0, left_bound - 5)

            right_bound = end
            while right_bound < num_points - 1 and linear_subtracted[right_bound] >= 0:
                right_bound += 1
            right_bound = min(num_points - 1, right_bound + 5)
            expanded_bounds.append([left_bound, right_bound])

        expanded_bounds.sort(key=lambda x: x[0])

        merged_bounds = []
        for current in expanded_bounds:
            if not merged_bounds:
                merged_bounds.append(current)
            else:
                prev_left, prev_right = merged_bounds[-1]
                current_left, current_right = current
                if current_left <= prev_right:
                    merged_bounds[-1][1] = max(prev_right, current_right)
                else:
                    merged_bounds.append(current)

        for left_bound, right_bound in merged_bounds:
            sub_slice = linear_subtracted[left_bound:right_bound + 1]
            freq_slice = self.freqs[left_bound:right_bound + 1]

            if len(sub_slice) == 0:
                continue

            area = np.trapz(sub_slice, freq_slice)

            sum_amp = np.sum(sub_slice)
            if sum_amp > 0:
                mean_freq = np.sum(freq_slice * sub_slice) / sum_amp
                variance = np.sum(((freq_slice - mean_freq) ** 2) * sub_slice) / sum_amp
                sigma = np.sqrt(max(0.0, variance))
            else:
                mean_freq = np.mean(freq_slice)
                sigma = np.std(freq_slice)

            peak_local_idx = np.argmax(sub_slice)
            freq_center = freq_slice[peak_local_idx]

            final_peaks_info.append({
                'freq_center': freq_center,
                'area': area,
                'mean': mean_freq,
                'sigma': sigma,
                'bounds': (freq_slice[0], freq_slice[-1])
            })

            region_marker = pg.LinearRegionItem(
                values=[freq_slice[0], freq_slice[-1]],
                brush=pg.mkBrush(255, 165, 0, 40), 
                pen=pg.mkPen('orange', width=1, style=Qt.DotLine),
                movable=False
            )
            self.bottom_plot.addItem(region_marker)
            self.peak_region_plots.append(region_marker)

        # Step 3: Populate Results Table（修改：加入当前Frame号的数据展示）
        self.table_widget.setRowCount(0)
        self.table_widget.setRowCount(len(final_peaks_info))
        
        current_frame = self.current_frame_idx
        
        for row, info in enumerate(final_peaks_info):
            self.table_widget.setItem(row, 0, QTableWidgetItem(f"{current_frame}"))
            self.table_widget.setItem(row, 1, QTableWidgetItem(f"{info['freq_center']:.4e}"))
            self.table_widget.setItem(row, 2, QTableWidgetItem(f"{info['area']:.4e}"))
            self.table_widget.setItem(row, 3, QTableWidgetItem(f"{info['mean']:.4e}"))
            self.table_widget.setItem(row, 4, QTableWidgetItem(f"{info['sigma']:.4e}"))

            # NEW: 将本次计算结果同步追加保存至内部内存数组，供后期生成 .csv 文件
            self.saved_csv_data.append([
                current_frame,
                info['freq_center'],
                info['area'],
                info['mean'],
                info['sigma']
            ])
            
        # 记录这次已经被成功点选并存入的 Frame 号
        self.saved_frames_set.add(current_frame)
        # 更新指示标签显示的 Frames 数量
        self.lbl_file_count.setText(f"Writted {len(self.saved_frames_set)} Frames")

        self.statusBar.showMessage(f"Found {len(final_peaks_info)} peak regions. Total recorded frames: {len(self.saved_frames_set)}", 4000)

    # NEW: 清空内部保存结果的槽函数
    def on_clear_csv(self):
        if not self.saved_csv_data:
            QMessageBox.information(self, "Notice", "No saved data to clear.")
            return
            
        reply = QMessageBox.question(
            self, "Clear Data", "Are you sure you want to clear all stored peak rows?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.saved_csv_data.clear()
            self.saved_frames_set.clear()
            self.lbl_file_count.setText("Writted 0 Frames")
            self.statusBar.showMessage("Stored data cleared completely.", 3000)

    # NEW: 选择路径并保存为 .csv 的槽函数
    def on_save_csv(self):
        if not self.saved_csv_data:
            QMessageBox.warning(self, "No Data", "There is no peak records to save! Please 'Find Peaks' first.")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Peak Findings CSV", self.current_dir, "CSV Files (*.csv)")
        if file_path:
            try:
                import csv
                with open(file_path, mode='w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    # 写入表头，首列加入 #Frame 用于区分不同帧
                    writer.writerow(["#Frame", "Freq Center", "Area", "Mean (1st)", "Sigma (2nd)"])
                    # 写入存储的所有历史波峰行数据
                    writer.writerows(self.saved_csv_data)
                    
                self.statusBar.showMessage(f"Successfully exported data to: {file_path}", 5000)
                QMessageBox.information(self, "Success", f"CSV file has been saved successfully.\nTotal Frames: {len(self.saved_frames_set)}")
            except Exception as e:
                QMessageBox.critical(self, "Save Failed", f"Failed to save CSV file:\n{str(e)}")

    # ==============================================================================
    # Core Plot Rendering & Hover Metadata Coordinate Extraction
    # ==============================================================================
    def on_select_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory", self.current_dir)
        if dir_path:
            self.current_dir = dir_path
            self.populate_file_list()
            self.statusBar.showMessage(f"Directory changed: {dir_path}", 4000)

    def update_slice_plots(self):
        if self.psd_arrays is None or self.freqs is None:
            return
        
        idx = self.current_frame_idx
        raw_slice = self.psd_arrays[idx, :]
        
        self.bottom_plot.setTitle(f"Result (PSD - Baseline) - #Frame {idx}")
        self.lbl_peak_frame.setText(f"#Frame {idx}")
        
        is_log_scale = self.radio_log.isChecked()
        
        if is_log_scale:
            floor_safe_raw = np.where(raw_slice <= 0, 1e-16, raw_slice)
            y_raw = np.log10(floor_safe_raw)
            self.top_plot.setLabel('left', 'Intensity (Log10)')
        else:
            y_raw = raw_slice
            self.top_plot.setLabel('left', 'Intensity (Linear)')
            
        self.curve_raw.setData(self.freqs[:-1], y_raw)
        
        if self.global_baseline is not None:
            if is_log_scale:
                floor_safe_base = np.where(self.global_baseline <= 0, 1e-16, self.global_baseline)
                y_base = np.log10(floor_safe_base)
            else:
                y_base = self.global_baseline
            self.curve_base.setData(self.freqs[:-1], y_base)
            
            subtracted = raw_slice - self.global_baseline
            if is_log_scale:
                floor_safe_sub = np.where(subtracted <= 0, 1e-16, subtracted)
                y_sub = np.log10(floor_safe_sub)
                self.bottom_plot.setLabel('left', 'Subtraction (Log10)')
            else:
                y_sub = subtracted
                self.bottom_plot.setLabel('left', 'Subtraction (Linear)')
            self.curve_sub.setData(self.freqs[:-1], y_sub)
        else:
            nan_arr = np.full_like(self.freqs[:-1], np.nan)
            self.curve_base.setData(self.freqs[:-1], nan_arr)
            self.curve_sub.setData(self.freqs[:-1], nan_arr)

    def on_waterfall_mouse_moved(self, pos):
        if self.psd_arrays is None or self.freqs is None or self.times is None:
            return
        
        vb = self.waterfall_plot.vb
        if vb.sceneBoundingRect().contains(pos):
            mouse_point = vb.mapSceneToView(pos)
            x_val = mouse_point.x()
            y_val = mouse_point.y()

            num_frames, num_freqs = self.psd_arrays.shape
            frame_idx = max(0, min(num_frames - 1, int(round(y_val))))
            
            freq_idx = np.searchsorted(self.freqs, x_val)
            freq_idx = max(0, min(num_freqs - 1, freq_idx))
            
            z_intensity = self.psd_arrays[frame_idx, freq_idx]
            self.statusBar.showMessage(
                f"Waterfall -> Freq: {x_val:.4e} Hz | Frame Index: {frame_idx} | "
                f"Time Code: {self.times[frame_idx]:.3f}s | PSD Intensity: {z_intensity:.4e}"
            )

    def on_top_mouse_moved(self, pos):
        if self.freqs is None:
            return
        vb = self.top_plot.plotItem.vb
        if vb.sceneBoundingRect().contains(pos):
            mouse_point = vb.mapSceneToView(pos)
            x_val = mouse_point.x()
            y_val = mouse_point.y()
            self.statusBar.showMessage(f"Top Slice Plot -> Freq: {x_val:.4e} Hz | Y-Axis: {y_val:.4e}")

    def on_bottom_mouse_moved(self, pos):
        if self.freqs is None:
            return
        vb = self.bottom_plot.plotItem.vb
        if vb.sceneBoundingRect().contains(pos):
            mouse_point = vb.mapSceneToView(pos)
            x_val = mouse_point.x()
            y_val = mouse_point.y()
            self.statusBar.showMessage(f"Bottom Slice Plot -> Freq: {x_val:.4e} Hz | Subtraction (Y-Axis): {y_val:.4e}")


# ==============================================================================
# Initialization Point
# ==============================================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DataMonitorWindow()
    window.show()
    sys.exit(app.exec_())
