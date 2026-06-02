VisionGuard — Real-Time Driver Safety System Using Deep Learning

• Fine-tuned YOLOv8 on a custom 3,438-image Indian road dataset achieving mAP50 of 0.777, detecting Indian-
specific classes like autorickshaws and two-wheelers with 0.984 accuracy

• Replaced expensive LiDAR sensors with monocular depth estimation (DepthAnything v2) for real-time distance
prediction and Time-to-Collision calculation using only a single RGB camera
• Trained a PPO reinforcement learning agent for intelligent collision alerts and deployed a live Streamlit web app
supporting real-time phone camera input via IP Webcam


