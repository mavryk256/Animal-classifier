import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

class AnimalClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Phân Loại Động Vật - AI Classifier (CPU Mode)")
        self.root.geometry("1000x750")
        self.root.configure(bg='#f5f5f5')

        # Cấu hình
        self.IMG_SIZE = 64
        self.model = None
        self.class_names = None
        self.current_image_path = None

        # Load model & classes
        self.load_model_and_classes()

        # Tạo giao diện
        self.create_widgets()

    def load_model_and_classes(self):
        try:
            # Load model
            model_path = "model.h5"
            if not os.path.exists(model_path):
                raise FileNotFoundError("Không tìm thấy model.h5!")
            self.model = load_model(model_path)
            print("[INFO] Đã load model.h5")

            # Load class names
            classes_path = "class_names.npy"
            if not os.path.exists(classes_path):
                raise FileNotFoundError("Không tìm thấy class_names.npy!")
            self.class_names = np.load(classes_path, allow_pickle=True)
            print(f"[INFO] Đã load {len(self.class_names)} classes: {', '.join(self.class_names)}")

            # Kiểm tra stats (chỉ để xác nhận)
            stats_path = "normalization_stats.npy"
            if os.path.exists(stats_path):
                stats = np.load(stats_path, allow_pickle=True).item()
                print(f"[INFO] Normalization method: {stats.get('method', 'unknown')}")

        except Exception as e:
            messagebox.showerror("Lỗi khởi động", f"Không thể khởi động ứng dụng:\n{str(e)}")
            self.root.destroy()

    def create_widgets(self):
        # Header
        header = tk.Frame(self.root, bg='#2c3e50', height=90)
        header.pack(fill=tk.X)
        tk.Label(
            header, text="PHÂN LOẠI ĐỘNG VẬT",
            font=('Arial', 28, 'bold'), bg='#2c3e50', fg='white'
        ).pack(pady=25)

        # Main container
        main = tk.Frame(self.root, bg='#f5f5f5')
        main.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Left: Ảnh
        left = tk.LabelFrame(main, text="Ảnh Đầu Vào", font=('Arial', 12, 'bold'),
                             bg='white', fg='#2c3e50', relief=tk.RIDGE, bd=3)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        canvas_frame = tk.Frame(left, bg='white')
        canvas_frame.pack(padx=15, pady=15, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_frame, width=500, height=450,
                                bg='#ecf0f1', highlightthickness=2, highlightbackground='#bdc3c7')
        self.canvas.pack()

        self.canvas.create_text(
            250, 225,
            text="Chưa có ảnh\n\nNhấn 'Tải Ảnh' để bắt đầu",
            font=('Arial', 14), fill='#7f8c8d', tags="placeholder"
        )

        # Buttons
        btn_frame = tk.Frame(left, bg='white')
        btn_frame.pack(pady=15)

        btn_style = {'font': ('Arial', 11, 'bold'), 'width': 15, 'height': 2,
                     'relief': tk.RAISED, 'bd': 3, 'cursor': 'hand2'}

        tk.Button(btn_frame, text="Tải Ảnh", bg='#3498db', fg='white',
                  command=self.upload_image, **btn_style).pack(side=tk.LEFT, padx=5)

        self.predict_btn = tk.Button(btn_frame, text="Phân Loại", bg='#2ecc71', fg='white',
                                     command=self.predict_image, state=tk.DISABLED, **btn_style)
        self.predict_btn.pack(side=tk.LEFT, padx=5)

        tk.Button(btn_frame, text="Xóa", bg='#e74c3c', fg='white',
                  command=self.clear_all, **btn_style).pack(side=tk.LEFT, padx=5)

        # Right: Kết quả
        right = tk.LabelFrame(main, text="Kết Quả Phân Loại", font=('Arial', 12, 'bold'),
                              bg='white', fg='#2c3e50', relief=tk.RIDGE, bd=3)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Kết quả chính
        res_main = tk.Frame(right, bg='white')
        res_main.pack(pady=15)
        tk.Label(res_main, text="Kết Quả:", font=('Arial', 12, 'bold'),
                 bg='white', fg='#34495e').pack()
        self.pred_label = tk.Label(res_main, text="---", font=('Arial', 24, 'bold'),
                                   fg='#7f8c8d', bg='white')
        self.pred_label.pack(pady=5)

        # Độ tin cậy
        conf_frame = tk.Frame(right, bg='white')
        conf_frame.pack(pady=10)
        tk.Label(conf_frame, text="Độ tin cậy:", font=('Arial', 11), bg='white').pack(side=tk.LEFT)
        self.conf_label = tk.Label(conf_frame, text="---%", font=('Arial', 11, 'bold'),
                                   bg='white', fg='#27ae60')
        self.conf_label.pack(side=tk.LEFT, padx=5)

        # Progress bar
        self.progress = ttk.Progressbar(right, length=320, mode='determinate')
        self.progress.pack(pady=10)

        ttk.Separator(right, orient='horizontal').pack(fill='x', pady=15)

        # Chi tiết
        tk.Label(right, text="Chi Tiết Dự Đoán:", font=('Arial', 11, 'bold'),
                 bg='white').pack(pady=5)

        text_frame = tk.Frame(right, bg='white')
        text_frame.pack(padx=15, pady=5, fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.text_widget = tk.Text(
            text_frame, height=12, width=38, font=('Courier', 10),
            wrap=tk.WORD, yscrollcommand=scrollbar.set,
            relief=tk.SUNKEN, bd=2
        )
        self.text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.text_widget.yview)

        # Status bar
        self.status = tk.Label(self.root, text="Sẵn sàng", bd=1, relief=tk.SUNKEN,
                               anchor=tk.W, font=('Arial', 9))
        self.status.pack(fill=tk.X, side=tk.BOTTOM, ipady=3)

    def upload_image(self):
        path = filedialog.askopenfilename(
        title="Chọn ảnh",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        if not path:
            return

        # Kiểm tra file tồn tại + đọc được
        if not os.path.exists(path):
            messagebox.showerror("Lỗi", "File không tồn tại!")
            return

        try:
            test_img = cv2.imread(path)
            if test_img is None:
                raise ValueError("OpenCV không đọc được ảnh!")
            print(f"[DEBUG] Ảnh hợp lệ: {path}")
        except Exception as e:
            messagebox.showerror("Lỗi", f"File không phải ảnh hợp lệ:\n{e}")
            return
        
        try:
            img = Image.open(path)
            img.thumbnail((500, 450), Image.LANCZOS)
            self.photo = ImageTk.PhotoImage(img)

            self.canvas.delete("all")
            self.canvas.create_image(250, 225, image=self.photo)

            self.current_image_path = path
            self.predict_btn.config(state=tk.NORMAL)
            self.status.config(text=f"Đã tải: {os.path.basename(path)}")

            # Reset kết quả
            self.pred_label.config(text="---", fg='#7f8c8d')
            self.conf_label.config(text="---%")
            self.progress['value'] = 0
            self.text_widget.delete('1.0', tk.END)

        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể tải ảnh:\n{e}")

    def predict_image(self):
        if not self.current_image_path or not os.path.exists(self.current_image_path):
            messagebox.showwarning("Cảnh báo", "Ảnh không tồn tại hoặc chưa được chọn!")
            return

        try:
            self.status.config(text="Đang phân loại...")
            self.root.update()

            # === ĐỌC ẢNH VÀ KIỂM TRA ===
            img = cv2.imread(self.current_image_path)
            if img is None:
                raise ValueError(f"Không thể đọc ảnh: {self.current_image_path}\n"
                                "Kiểm tra: file có tồn tại? định dạng hợp lệ? quyền truy cập?")

            # Resize + chuẩn hóa
            img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
            img = img.astype("float32") / 255.0
            img = np.expand_dims(img, axis=0)

            # Dự đoán
            preds = self.model.predict(img, verbose=0)[0]
            max_idx = np.argmax(preds)
            max_conf = preds[max_idx] * 100

            # Threshold
            THRESHOLD = 40.0
            if max_conf < THRESHOLD:
                result = "Không rõ"
                color = '#e67e22'
            else:
                result = self.class_names[max_idx].upper()
                color = '#27ae60'

            # Hiển thị
            self.pred_label.config(text=result, fg=color)
            self.conf_label.config(text=f"{max_conf:.1f}%")
            self.progress['value'] = max_conf

            # Chi tiết
            self.text_widget.delete('1.0', tk.END)
            self.text_widget.insert(tk.END, "─" * 45 + "\n")
            self.text_widget.insert(tk.END, f"{'#':<3}{'Class':<18}{'Conf':<10}\n")
            self.text_widget.insert(tk.END, "─" * 45 + "\n")

            sorted_idx = np.argsort(preds)[::-1]
            for i, idx in enumerate(sorted_idx[:10], 1):
                name = self.class_names[idx]
                conf = preds[idx] * 100
                star = " ★" * min(i, 3)
                self.text_widget.insert(tk.END, f"{i:<3}{name:<18}{conf:>6.1f}%{star}\n")

            self.text_widget.insert(tk.END, "─" * 45 + "\n")
            self.status.config(text=f"Hoàn tất: {result} ({max_conf:.1f}%)")

        except Exception as e:
            messagebox.showerror("Lỗi phân loại", f"Không thể xử lý ảnh:\n{str(e)}")
            self.status.config(text="Lỗi!")

    def clear_all(self):
        self.canvas.delete("all")
        self.canvas.create_text(
            250, 225,
            text="Chưa có ảnh\n\nNhấn 'Tải Ảnh' để bắt đầu",
            font=('Arial', 14), fill='#7f8c8d', tags="placeholder"
        )
        self.current_image_path = None
        self.predict_btn.config(state=tk.DISABLED)
        self.pred_label.config(text="---", fg='#7f8c8d')
        self.conf_label.config(text="---%")
        self.progress['value'] = 0
        self.text_widget.delete('1.0', tk.END)
        self.status.config(text="Sẵn sàng")


def main():
    root = tk.Tk()
    try:
        root.iconbitmap('icon.ico')
    except:
        pass
    app = AnimalClassifierGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
