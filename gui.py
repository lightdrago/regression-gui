import sys
import random
import traceback
from pathlib import Path
from PySide6 import QtCore, QtWidgets, QtGui
import pandas as pd

import numpy.core._exceptions

from main import App as RegressionApp
from qt_pandas_tableview import PandasModel


class InputDataWidget(QtWidgets.QWidget):
    def __init__(self, regression_app):
        super().__init__()

        # data
        self.filename = None
        self.top_lines = None
        self.app = regression_app

        self.X_columns = []
        self.Y_column = None
        self.regressors = []

        self.results_widget = None

        self.threadpool = QtCore.QThreadPool()

        self.job_statuses = {True: 'Идут вычисления: {0}', False: 'Ожидание ввода'}

        self.setWindowTitle('Ввод данных')
        # layouts
        self.layout = QtWidgets.QVBoxLayout(self)
        self.file_selection_layout = QtWidgets.QHBoxLayout()
        self.dataframe_preview_layout = QtWidgets.QHBoxLayout()
        self.options_layout = QtWidgets.QHBoxLayout()
        self.action_layout = QtWidgets.QHBoxLayout()

        self.layout.addLayout(self.file_selection_layout)
        self.layout.addLayout(self.dataframe_preview_layout)
        self.layout.addLayout(self.options_layout)
        self.layout.addLayout(self.action_layout)

        # file selection
        layout = self.file_selection_layout
        
        self.dataset_file_dialog_invoke_button = QtWidgets.QPushButton("Выбрать файл...")
        self.dataset_filename_label = QtWidgets.QLabel("Файл не выбран")
        self.dataset_headers_checkbox = QtWidgets.QCheckBox("Заголовки")
        self.dataset_top_lines_checkbox = QtWidgets.QCheckBox("Загрузить первых:")
        self.dataset_top_lines_checkbox.stateChanged.connect(self.top_lines_checked)
        self.dataset_top_lines_spinbox = QtWidgets.QSpinBox()
        self.dataset_top_lines_spinbox.setMaximum(10000000)
        self.dataset_top_lines_spinbox.setMinimum(1)
        self.dataset_top_lines_spinbox.setValue(1000)
        self.dataset_top_lines_spinbox.setEnabled(False)
        self.dataset_reload_button = QtWidgets.QPushButton("Загрузить")
        
        layout.addWidget(self.dataset_file_dialog_invoke_button)
        layout.addWidget(self.dataset_filename_label)
        layout.addWidget(self.dataset_headers_checkbox)
        layout.addWidget(self.dataset_top_lines_checkbox)
        layout.addWidget(self.dataset_top_lines_spinbox)
        layout.addStretch()
        layout.addWidget(self.dataset_reload_button)
        
        self.dataset_file_dialog_invoke_button.clicked.connect(self.invoke_file_dialog)
        self.dataset_reload_button.clicked.connect(self.reload_dataset)

        # dataframe preview
        layout = self.dataframe_preview_layout

        self.dataframe_table = QtWidgets.QTableView()
        self.dataframe_model = None

        layout.addWidget(self.dataframe_table)

        # options
        layout = self.options_layout

        self.column_selection_layout = QtWidgets.QVBoxLayout()
        layout.addLayout(self.column_selection_layout)

        self.regression_selection_layout = QtWidgets.QVBoxLayout()
        layout.addLayout(self.regression_selection_layout)

        # columns
        layout = self.column_selection_layout

        layout.addWidget(QtWidgets.QLabel('Столбцы X'))

        self.X_columns_listbox = QtWidgets.QListWidget()
        self.X_columns_listbox.setSelectionMode(self.X_columns_listbox.MultiSelection)
        layout.addWidget(self.X_columns_listbox)

        layout.addWidget(QtWidgets.QLabel('Столбец Y'))

        self.Y_column_combobox = QtWidgets.QComboBox()
        layout.addWidget(self.Y_column_combobox)

        # regressors
        layout = self.regression_selection_layout

        layout.addWidget(QtWidgets.QLabel('Модели регрессии'))

        self.regressors_listbox = QtWidgets.QListWidget()
        self.regressors_listbox.setSelectionMode(self.regressors_listbox.MultiSelection)
        layout.addWidget(self.regressors_listbox)
        self.regressors_listbox.addItems(rg.title for rg in self.app.get_regressors())

        # actions
        layout = self.action_layout

        self.start_button = QtWidgets.QPushButton('Запустить')
        self.start_button.clicked.connect(self.run_regressors)
        self.job_status_label = QtWidgets.QLabel(self.job_statuses[False])

        layout.addStretch()
        layout.addWidget(self.job_status_label)
        layout.setAlignment(self.job_status_label, QtCore.Qt.AlignRight)
        layout.addWidget(self.start_button)
        layout.setAlignment(self.start_button, QtCore.Qt.AlignRight)
        

        
    @QtCore.Slot()
    def top_lines_checked(self):
        check_state = self.dataset_top_lines_checkbox.checkState()
        is_checked = check_state == check_state.Checked
        self.dataset_top_lines_spinbox.setEnabled(is_checked)
        self.top_lines = is_checked

    @QtCore.Slot()
    def invoke_file_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, caption="Выберите файл с данными", filter="Файлы в формате CSV (*.csv)")
        self.filename = Path(path)
        self.dataset_filename_label.setText(self.filename.name)
    
    @QtCore.Slot()
    def reload_dataset(self, skip_header_check=False):
        if self.filename:
            has_headers = self.dataset_headers_checkbox.checkState()
            top_lines = self.dataset_top_lines_spinbox.value() if self.top_lines else 0
            self.app.load_data(self.filename, has_headers, top_lines=top_lines)
            self.dataframe_model = PandasModel(self.app.df_preview)
            if not has_headers:
                self.dataframe_model.setHorizontalOverride([i for i in range(self.app.df_preview._data.shape[1])])
            self.dataframe_table.setModel(self.dataframe_model)

            columns = [str(column) for column in self.app.df_preview.columns]
            self.X_columns_listbox.clear()
            self.X_columns_listbox.addItems(columns)
            self.Y_column_combobox.clear()
            self.Y_column_combobox.addItems(columns)

            if (not skip_header_check
            and all(self.app.df.iloc[0].apply(lambda item: isinstance(item, str)))
            and all(self.app.df.iloc[1].apply(lambda item: isinstance(item, str)))):
                if QtWidgets.QMessageBox.question(self, 'Заголовки', 'Возможно, первая строка файла содержит заголовки. Исправить?',
                    buttons=QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                    defaultButton=QtWidgets.QMessageBox.Yes
                ) == QtWidgets.QMessageBox.Yes:
                    self.dataset_headers_checkbox.setCheckState(QtCore.Qt.CheckState.Checked)
                    # self.dataset_reload_button.click()
                    self.reload_dataset(skip_header_check=True)

    
    @QtCore.Slot()
    def run_regressors(self):
        self.X_columns = [index.row() for index in self.X_columns_listbox.selectedIndexes()[:]]
        self.Y_column = self.Y_column_combobox.currentIndex()
        self.regressors = [index.row() for index in self.regressors_listbox.selectedIndexes()[:]]

        if len(self.X_columns) == 0:
            QtWidgets.QMessageBox.critical(self, 'Ошибка', 'Вы не выбрали ни одного столбца входных данных.')
            return
        
        if len(self.regressors) == 0:
            QtWidgets.QMessageBox.critical(self, 'Ошибка', 'Вы не выбрали ни одной модели регрессии.')
            return
        
        if self.Y_column in self.X_columns:
            QtWidgets.QMessageBox.critical(self, 'Ошибка', f'Вы выбрали столбец {str(self.app.df.columns[self.Y_column])} в качестве входного и в качестве выходного.')
            return
        
        self.app.set_X(self.X_columns)
        self.app.set_Y(self.Y_column)
        self.app.set_regressors(self.regressors)

        non_numeric_columns = []
        for column in self.app.X.columns:
            if not pd.api.types.is_numeric_dtype(self.app.X[column]):
                non_numeric_columns.append(column)
        
        if len(non_numeric_columns):
            QtWidgets.QMessageBox.critical(
                self, 'Ошибка', 
                f'Следующие столбцы содержат нечисловые данные и не могут быть использованы в качестве Х:\n'
                f'{";".join([str(c) for c in non_numeric_columns])}'
            )
            return

        self.start_button.setEnabled(False)
        self.job_status_label.setText(self.job_statuses[True].format('подготовка...'))

        # async stuff
        worker = RegressionAppWorker(self.app)
        worker.signals.finished.connect(self.regressors_finished)
        worker.signals.error.connect(self.regressors_errored)
        self.threadpool.start(worker)

    @QtCore.Slot()
    def regressors_finished(self):
        self.job_status_label.setText(self.job_statuses[False])

        self.results_widget = ResultsWidget(self.app)
        self.results_widget.setWindowModality(QtCore.Qt.ApplicationModal)
        self.results_widget.resize(900, 900)
        self.results_widget.show()
        self.start_button.setEnabled(True)
    
    @QtCore.Slot()
    def regressors_errored(self, exception_data):
        exc_info, readable = exception_data
        if readable:
            QtWidgets.QMessageBox.critical(self, 'Ошибка', f'{readable}.')
        else:
            QtWidgets.QMessageBox.critical(self, 'Непредвиденная ошибка', f'Возникла непредвиденная ошибка!\n{exc_info}')
    
    @QtCore.Slot()
    def regressors_progress(self, current_info):
        self.job_status_label.setText(self.job_statuses[True].format(current_info))



class ResultsWidget(QtWidgets.QWidget):
    def __init__(self, regression_app):
        super().__init__()

        self.setWindowTitle('Результаты')

        self.app = regression_app

        self.layout = QtWidgets.QVBoxLayout(self)

        self.tab_control = QtWidgets.QTabWidget(self)
        
        for regressor in self.app.enabled_regressors:
            plot_file = None
            tab_text = "???"

            if regressor.title in self.app.last_results.keys():
                tab_text = str(self.app.last_results[regressor.title])
                if regressor.title in self.app.last_plots.keys():
                    plot_file = self.app.last_plots[regressor.title]
            else:
                tab_text = "Вычисления завершились с ошибкой."
            tab_body = TextBoxResultsWidget(tab_text, plot_file)
            self.tab_control.addTab(tab_body, regressor.title)
        
        self.layout.addWidget(self.tab_control)

class TextBoxResultsWidget(QtWidgets.QWidget):
    def __init__(self, text, plot_file=None):
        super().__init__()

        font = QtGui.QFont("Monospace")
        font.setStyleHint(QtGui.QFont.Monospace)

        self.layout = QtWidgets.QVBoxLayout(self)

        if plot_file:
            self.pic = QtGui.QImage(plot_file)
            self.pixmap = QtGui.QPixmap(self.pic)
            self.plot_label = QtWidgets.QLabel()
            self.plot_label.setPixmap(self.pixmap)
            self.layout.addWidget(self.plot_label)

        self.text_box = QtWidgets.QPlainTextEdit(text)
        self.text_box.setReadOnly(True)
        self.text_box.setFont(font)
        self.layout.addWidget(self.text_box)

    
class RegressionAppWorker(QtCore.QRunnable):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.signals = RegressorAppWorkerSignals()

    @QtCore.Slot()
    def run(self):
        self.app.last_plots.clear()
        self.app.last_results.clear()
        try:
            for r in self.app.enabled_regressors:
                try:
                    self.signals.progress.emit(r.title)
                    r.run(self.app.X, self.app.Y)
                    self.app.last_results[r.title] = r.get_stats(self.app.Y, self.app.X)
                    r.save_summary(self.app.Y, self.app.X)
                    if self.app.X.shape[1] == 1:
                        self.app.last_plots[r.title] = r.save_plot_image(self.app.Y, self.app.X)
                except numpy.core._exceptions._ArrayMemoryError as e:
                    self.signals.error.emit((traceback.format_exc(), f'Недостаточно памяти для хранения данных (требуется {e._size_to_string(e._total_size)})',))
                except Exception as e:
                    traceback.print_exc()
                    self.signals.error.emit((traceback.format_exc(), None))
        except Exception as e:
            traceback.print_exc()
            self.signals.error.emit((traceback.format_exc(), None))
        finally:
            self.signals.finished.emit()


class RegressorAppWorkerSignals(QtCore.QObject):
    finished = QtCore.Signal()
    error = QtCore.Signal(tuple)
    progress = QtCore.Signal(str)
    



if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    regression_app = RegressionApp()
    widget = InputDataWidget(regression_app)
    widget.resize(800, 600)
    widget.show()

    sys.exit(app.exec_())