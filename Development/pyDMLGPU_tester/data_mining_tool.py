from tkinter import *
from tkinter import filedialog
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from locals import *
from worker import Worker
import ctypes
import time


# Pick file which contains database information to be processed by the parallel algorithms
def choose_database():
    try:
        database_path = filedialog.askopenfilename(initialdir="\\", title="Select file",
                                                   filetypes=(("all files", "*.*"),))
        # set database path to worker_thread
        worker_thread.set_database_file_path(database_path)

        file_name = database_path.split('/')[-1]
        database_chosen.configure(text=file_name)

        result_box.config(state=NORMAL)
        result_box.insert(END, BOX_POINTER + 'Database file selected ...' + NEW_LINE, STANDARD_FONT_COLOR)
        result_box.config(state=DISABLED)
    except FileNotFoundError:
        print('FileNotFoundError:: action skipped')


def choose_names_file():
    try:
        names_path = filedialog.askopenfilename(initialdir="\\", title="Select file",
                                                filetypes=(("all files", "*.*"),))
        worker_thread.set_names(names_path)

        file_name = names_path.split('/')[-1]
        names_chosen.configure(text=file_name)

        result_box.config(state=NORMAL)
        result_box.insert(END, BOX_POINTER + 'Names database selected ...' + NEW_LINE, STANDARD_FONT_COLOR)
        result_box.config(state=DISABLED)
    except FileNotFoundError:
        print('FileNotFoundError:: action skipped')


# Launch thread which handles the back-end action
def on_start():
    # set algorithm type
    worker_thread.set_algorithm_type(alg_value.get())
    # set minimum support
    try:
        min_supp = float(min_supp_entry.get())
        worker_thread.set_minimum_support(min_supp)
    except ValueError:
        result_box.config(state=NORMAL)
        result_box.insert(END, BOX_POINTER + ERROR_MIN_SUPP_VALUE, ERROR_FONT_COLOR)
        result_box.config(state=DISABLED)
        return
    # request processing
    worker_thread.request_processing()


def on_close():
    worker_thread.kill()
    tk.destroy()


def clean_console():
    worker_thread.stop()

    result_box.config(state=NORMAL)
    result_box.delete(1.0, END)
    result_box.insert(END, READY_LOG, STANDARD_FONT_COLOR)
    result_box.config(state=DISABLED)


def clear_database():
    worker_thread.free_database()
    database_chosen.configure(text='Database: ')
    result_box.config(state=NORMAL)
    result_box.insert(END, BOX_POINTER + 'Database cleared ...' + NEW_LINE, STANDARD_FONT_COLOR)
    result_box.config(state=DISABLED)


def clear_names():
    worker_thread.free_names()
    names_chosen.configure(text='Names: ')
    result_box.config(state=NORMAL)
    result_box.insert(END, BOX_POINTER + 'Names cleared ...' + NEW_LINE, STANDARD_FONT_COLOR)
    result_box.config(state=DISABLED)


if __name__ == '__main__':
    # set dpi awareness
    awareness = ctypes.c_int()
    errorCode = ctypes.windll.shcore.GetProcessDpiAwareness(0, ctypes.byref(awareness))
    errorCode = ctypes.windll.shcore.SetProcessDpiAwareness(2)
    success = ctypes.windll.user32.SetProcessDPIAware()

    # root
    tk = Tk()
    tk.resizable(False, False)
    tk.title(APP_TITLE)
    tk.iconbitmap(r'resources\icon_app.ico')
    tk.configure(background=BACKGROUND_FRAME_COLOR)

    # style
    style = ttk.Style()
    style.configure('TLabel', background=BACKGROUND_FRAME_COLOR)
    style.configure('TLabelframe', background=BACKGROUND_FRAME_COLOR)
    style.configure('TFrame', background=BACKGROUND_FRAME_COLOR)
    style.configure('TLabelframe.Label', background=BACKGROUND_FRAME_COLOR)
    style.configure('TButton', background=BACKGROUND_FRAME_COLOR)
    style.configure('TOptionmenu', background=BACKGROUND_FRAME_COLOR)

    title_label = ttk.Label(tk, text=APP_TITLE, font=('Arial', 20, 'bold'))
    title_label.pack(padx=10, pady=10)

    # containing frame - top
    top_frame = ttk.Frame(tk)
    top_frame.pack()

    # settings label frame
    settings_frame = ttk.LabelFrame(top_frame, text='Settings', padding=5)
    settings_frame.pack(padx=10, pady=10, side=LEFT)

    database_chosen = ttk.Label(settings_frame, text='Database: ')
    database_chosen.grid(row=0, column=0, padx=5, pady=5)
    database_chosen.configure(width=10)

    # filedialog here on row=0, column=1, or something like this
    database_chooser_button = ttk.Button(settings_frame, text='Select Database File', command=choose_database)
    database_chooser_button.grid(row=0, column=1, padx=5, pady=5)

    names_chosen = ttk.Label(settings_frame, text='Names: ')
    names_chosen.grid(row=1, column=0, padx=5, pady=5)
    names_chosen.configure(width=10)

    names_chooser_button = ttk.Button(settings_frame, text='Select Names File', command=choose_names_file)
    names_chooser_button.grid(row=1, column=1, padx=5, pady=5)

    min_supp_label = ttk.Label(settings_frame, text='minimum support')
    min_supp_label.grid(row=2, column=0, padx=5, pady=5)

    # min supp entry box
    min_supp_entry = ttk.Entry(settings_frame, width=10)
    min_supp_entry.grid(row=2, column=1, padx=5, pady=5)

    # processing label frame
    processing_frame = ttk.LabelFrame(top_frame, text='Processing Frame', padding=5)
    processing_frame.pack(padx=10, pady=10, side=LEFT)

    # algorithm chooser
    alg_value = StringVar(processing_frame)
    alg_value.set(CHOSE_ALGORITHM)
    alg_chooser = ttk.OptionMenu(processing_frame, alg_value, CHOSE_ALGORITHM, CHOSE_ALGORITHM,
                                 PARALLEL_APRIORI_ALGORITHM, PARALLEL_MINER_ALGORITHM)
    alg_chooser.configure(width=17)
    alg_chooser.grid(row=0, column=0, columnspan=2, padx=5, pady=5)

    # start button
    start_button = ttk.Button(processing_frame, text='Start', width=15, command=on_start)
    start_button.grid(row=1, column=0, padx=5, pady=5)

    # clean button
    clean_button = ttk.Button(processing_frame, text='Clean Console', width=15, command=clean_console)
    clean_button.grid(row=1, column=1, padx=5, pady=5)

    # clear databases fields
    clear_database_field = ttk.Button(processing_frame, text='Clear Database', width=15, command=clear_database)
    clear_database_field.grid(row=3, column=0, padx=5, pady=5)

    clear_names_field = ttk.Button(processing_frame, text='Clear Names', width=15, command=clear_names)
    clear_names_field.grid(row=3, column=1, padx=5, pady=5)

    # result box; global is used to be able to write in result box from outer scope
    result_box = ScrolledText(tk, bg='black', width=100, height=40)
    result_box.pack(padx=10, pady=10)

    result_box.insert(END, READY_LOG, STANDARD_FONT_COLOR)

    # don't let user write in result box
    result_box.config(state=DISABLED)

    # colored text styles for result_box
    result_box.tag_config(STANDARD_FONT_COLOR, foreground=STANDARD_COLOR)
    result_box.tag_config(RESULTS_FONT_COLOR, foreground=RESULT_COLOR)
    result_box.tag_config(ERROR_FONT_COLOR, foreground=ERROR_COLOR)

    # create the thread worker
    worker_thread = Worker(tk, result_box)
    worker_thread.start()

    # handle the closing of GUI
    tk.protocol("WM_DELETE_WINDOW", on_close)

    mainloop()
