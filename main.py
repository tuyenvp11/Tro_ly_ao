import tkinter as tk
from tkinter import scrolledtext
from gtts import gTTS
import playsound
import os
import threading
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import speech_recognition as sr

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier

# Hàm phát giọng nói
def tra_loi_giong_noi(text):
    tts = gTTS(text=text, lang='vi')
    filename = "../tra_loi.mp3"
    tts.save(filename)
    playsound.playsound(filename)
    os.remove(filename)


# Hàm để nghe và chuyển đổi giọng nói thành văn bản
def nghe_giong_noi():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        gui_output("Đang lắng nghe...")
        audio = recognizer.listen(source)
    try:
        # Sử dụng Google Speech Recognition để chuyển âm thanh thành văn bản
        text = recognizer.recognize_google(audio, language='vi-VN')
        gui_output(f"Bạn: {text}")
        return text
    except sr.UnknownValueError:
        gui_output("Xin lỗi, tôi không hiểu bạn nói gì.")
        return None
    except sr.RequestError:
        gui_output("Có lỗi xảy ra với dịch vụ nhận dạng giọng nói.")
        return None


# Hàm đầu ra giao diện
def gui_output(text):
    chat_window.configure(state='normal')
    chat_window.insert(tk.END, text + '\n')
    chat_window.configure(state='disabled')
    chat_window.yview(tk.END)


# Đọc dữ liệu từ dataset
def load_data(file_path):
    data = pd.read_csv(file_path, encoding='utf-8')
    # Đặt tên cột cho dữ liệu
    data.columns = ['Câu hỏi', 'Câu trả lời']
    data.dropna(inplace=True)
    return data['Câu hỏi'], data['Câu trả lời']


# Huấn luyện mô hình học máy
def train_model(questions, answers):
    # Sử dụng TF-IDF để chuyển câu hỏi thành vector
    vector = TfidfVectorizer()
    x = vector.fit_transform(questions)

    # Huấn luyện mô hình SVM
    svm_model = SVC(kernel='linear', probability=True)

    # Huấn luyện mô hình Naive Bayes
    nb_model = MultinomialNB()

    # Kết hợp cả hai mô hình bằng Voting Classifier
    model = VotingClassifier(estimators=[('svm', svm_model), ('nb', nb_model)], voting='soft')
    model.fit(x, answers)

    return model, vector


# Bước 3: Dự đoán câu trả lời dựa trên câu hỏi
def predict_answer(model, vector, question):
    x_test = vector.transform([question])
    prediction = model.predict(x_test)
    return prediction[0]

# Hàm xử lý gửi câu hỏi
def send_message():
    user_message = user_input.get("1.0", tk.END).strip()
    if user_message:
        gui_output(f"Bạn: {user_message}")
        user_input.delete("1.0", tk.END)

        # Kiểm tra nếu người dùng nói tạm biệt
        if "tạm biệt" in user_message.lower() or "chào tạm biệt" in user_message.lower():
            gui_output("Trợ lý ảo: Tạm biệt! Hẹn gặp lại.")
            tra_loi_giong_noi("Tạm biệt! Hẹn gặp lại.")
            root.quit()
            return

        # Dự đoán câu trả lời
        answer = predict_answer(model, vector, user_message)
        gui_output(f"Trợ lý ảo: {answer}")
        tra_loi_giong_noi(answer)


# Hàm bắt đầu nghe giọng nói và trả lời
def start_listening():
    question = nghe_giong_noi()
    if question is not None:
        # Kiểm tra nếu người dùng nói tạm biệt
        if "tạm biệt" in question.lower() or "chào tạm biệt" in question.lower():
            gui_output("Trợ lý ảo: Tạm biệt! Hẹn gặp lại.")
            tra_loi_giong_noi("Tạm biệt! Hẹn gặp lại.")
            root.quit()
            return

        # Dự đoán câu trả lời
        answer = predict_answer(model, vector, question)
        gui_output(f"Trợ lý ảo: {answer}")
        tra_loi_giong_noi(answer)


# Tạo giao diện
def create_chat_gui():
    global chat_window, user_input, root
    root = tk.Tk()
    root.title("Trợ Lý Ảo")
    # Tạo khung chat
    chat_frame = tk.Frame(root)
    chat_frame.pack(padx=10, pady=10)
    # Cửa sổ hiển thị nội dung chat
    chat_window = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, width=60, height=20, state='disabled',
                                            font=("Arial", 12))
    chat_window.pack()
    # Khung nhập liệu cho người dùng
    user_input = tk.Text(root, wrap=tk.WORD, width=50, height=3, font=("Arial", 12))
    user_input.pack(pady=10)

    # Khung nút điều khiển
    control_frame = tk.Frame(root)
    control_frame.pack()
    # Nút gửi tin nhắn
    send_button = tk.Button(control_frame, text="Gửi", command=send_message, font=("Arial", 12))
    send_button.pack(side=tk.LEFT, padx=10)
    # Nút nhận giọng nói
    listen_button = tk.Button(control_frame, text="Nhấn để nói",
                              command=lambda: threading.Thread(target=start_listening).start(), font=("Arial", 12))
    listen_button.pack(side=tk.LEFT, padx=10)
    # Chạy giao diện
    root.mainloop()


if __name__ == "__main__":
    # Đọc dữ liệu và huấn luyện mô hình
    questions, answers = load_data('dataset_information_tech.csv')
    model, vector = train_model(questions, answers)
    create_chat_gui()
