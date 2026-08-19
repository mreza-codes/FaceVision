# FaceVision — Real-time Face Recognition  
FaceVision is a real-time face recognition system built using powerful deep-learning models to detect and identify faces with high accuracy. The system uses ArcFace for generating precise face embeddings and YOLO for fast and reliable face detection. It includes a clean and simple PyQt interface, a custom face database, and a modular architecture that makes it easy to extend, modify, and integrate into larger applications.

FaceVision captures live video frames, detects faces in real time, extracts a unique numerical embedding for each detected face, and compares these embeddings with stored data in the local database. This enables the system to identify individuals quickly and accurately. The Add-Person workflow allows users to register new identities by capturing a face, generating its embedding, and saving it for future recognition. This makes FaceVision suitable for attendance systems, access control, smart home applications, and academic or industrial research projects.

The user interface is designed to be simple and intuitive. Users can start the camera, stop it, add new individuals, and view recognition results instantly. The main display area shows the live camera feed, while the control buttons allow easy interaction with the system. The modular design of the codebase allows developers to replace models, adjust thresholds, add new features, or integrate cloud-based storage solutions without difficulty.

FaceVision is also an educational resource for those interested in computer vision and machine learning. It demonstrates how face detection, embedding generation, similarity comparison, and graphical interface design can be combined to create a functional and efficient real-time recognition system. The project structure is organized clearly to help developers understand and extend the system easily.

Features:  
- Real-time face detection & recognition  
- ArcFace embeddings for high-precision identity representation  
- YOLO-based face detection  
- Add-person database for registering new identities  
- PyQt user interface  
- Modular and extendable architecture  

Project Structure:  
faces/  
models/  
models/models/  
main.py  
ui_facevision.py  
facevision.ui  
icon.ico  

Run:  
python main.py

این سامانه یک سیستم تشخیص چهره‌ی لحظه‌ای مبتنی بر هوش مصنوعی است که توانایی شناسایی افراد را در تصویر و ویدئو دارد. رابط کاربری ساده‌ای برای مدیریت دوربین، افزودن افراد جدید و مشاهده‌ی نتیجه‌ی تشخیص ارائه می‌دهد و به‌دلیل سرعت بالا و دقت مناسب، برای استفاده در کاربردهای عملی و محیط‌های واقعی بسیار مناسب است.
