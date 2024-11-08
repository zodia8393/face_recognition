import cv2
import numpy as np
from deepface import DeepFace
import mysql.connector
from mysql.connector import Error
import logging
import configparser
import os
import threading
from queue import Queue, Empty
import json
from typing import List, Dict, Any, Tuple
from collections import deque
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import pyttsx3
import time

# TensorFlow GPU 설정
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# TensorFlow 경고 메시지 숨기기
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 설정 파일 로드
config = configparser.ConfigParser()
config_path = 'C:/Users/BioBrain/Desktop/WS/WORK/face_recognition_system/config.ini'

class ConfigManager:
    @staticmethod
    def load_config():
        if not os.path.exists(config_path):
            ConfigManager.create_default_config()
        else:
            try:
                config.read(config_path, encoding='utf-8')
            except configparser.Error as e:
                logging.error(f"설정 파일 읽기 오류: {e}")
                ConfigManager.create_default_config()

    @staticmethod
    def create_default_config():
        config['Database'] = {
            'Host': 'localhost',
            'User': 'root',
            'Password': '8393',
            'Port': '3300',
            'Database': 'face_recognition_db'
        }
        config['Logging'] = {
            'LogFile': 'face_recognition.log',
            'Level': 'INFO'
        }
        config['FaceRecognition'] = {
            'DetectionBackend': 'opencv',
            'RecognitionModel': 'Facenet',
            'DistanceThreshold': '0.6',
            'ResultSmoothingFrames': '10'
        }
        with open(config_path, 'w', encoding='utf-8') as configfile:
            config.write(configfile)
        logging.info("기본 설정 파일 생성됨")

ConfigManager.load_config()

# 로깅 설정
log_file = config.get('Logging', 'LogFile', fallback='face_recognition.log')
log_level = config.get('Logging', 'Level', fallback='INFO')
logging.basicConfig(
    filename=log_file,
    level=getattr(logging, log_level),
    format='%(asctime)s:%(levelname)s:%(message)s',
    encoding='utf-8'
)

class DatabaseManager:
    def __init__(self):
        self.connection = None
        self.create_connection()
        self.create_database()
        self.create_users_table()
        self.users_cache = {}
        self.last_cache_update = 0
        self.cache_ttl = 60  # 캐시 유효 시간 (초)

    def create_connection(self):
        try:
            self.connection = mysql.connector.connect(
                host=config.get('Database', 'Host'),
                user=config.get('Database', 'User'),
                password=config.get('Database', 'Password'),
                port=config.getint('Database', 'Port'),
                database=config.get('Database', 'Database')
            )
            logging.info("MySQL 데이터베이스에 연결됨")
        except Error as e:
            logging.error(f"MySQL 데이터베이스 연결 오류: {e}")
            raise

    def create_database(self):
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(f"CREATE DATABASE IF NOT EXISTS {config.get('Database', 'Database')} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            self.connection.commit()
            logging.info(f"데이터베이스 '{config.get('Database', 'Database')}' 생성 완료 또는 이미 존재함")
        except Error as e:
            logging.error(f"데이터베이스 생성 오류: {e}")
            raise

    def create_users_table(self):
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        name VARCHAR(255) NOT NULL,
                        embedding JSON NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        CONSTRAINT uc_name UNIQUE (name)
                    ) ENGINE=InnoDB
                """)
                cursor.execute("""
                    SELECT COUNT(1) IndexIsThere FROM INFORMATION_SCHEMA.STATISTICS
                    WHERE table_schema=DATABASE() AND table_name='users' AND index_name='idx_name'
                """)
                index_exists = cursor.fetchone()[0]
                if not index_exists:
                    cursor.execute("CREATE INDEX idx_name ON users (name)")
                    logging.info("사용자 테이블에 'idx_name' 인덱스 생성됨")
            self.connection.commit()
            logging.info("'users' 테이블 생성 완료 또는 이미 존재함")
        except Error as e:
            logging.error(f"사용자 테이블 생성 오류: {e}")
            raise

    def register_new_face(self, name: str, embedding: np.ndarray) -> bool:
        try:
            with self.connection.cursor() as cursor:
                query = "INSERT INTO users (name, embedding) VALUES (%s, %s)"
                embedding_json = json.dumps(embedding.tolist())
                cursor.execute(query, (name, embedding_json))
            self.connection.commit()
            logging.info(f"새로운 얼굴 등록: {name}")
            self.invalidate_cache()
            return True
        except Error as e:
            logging.error(f"새로운 얼굴 등록 오류: {e}")
            self.connection.rollback()
            return False

    def user_exists(self, name: str) -> bool:
        try:
            with self.connection.cursor() as cursor:
                query = "SELECT COUNT(*) FROM users WHERE name = %s"
                cursor.execute(query, (name,))
                count = cursor.fetchone()[0]
                return count > 0
        except Error as e:
            logging.error(f"사용자 존재 여부 확인 오류: {e}")
            return False

    def get_all_users_info(self) -> List[Dict[str, Any]]:
        try:
            with self.connection.cursor(dictionary=True) as cursor:
                cursor.execute("SELECT id, name, created_at FROM users")
                return cursor.fetchall()
        except Error as e:
            logging.error(f"사용자 정보 가져오기 오류: {e}")
            return []

    def update_user_name(self, user_id: int, new_name: str) -> bool:
        try:
            with self.connection.cursor() as cursor:
                query = "UPDATE users SET name = %s WHERE id = %s"
                cursor.execute(query, (new_name, user_id))
            self.connection.commit()
            logging.info(f"사용자 이름 업데이트: ID {user_id}, 새 이름 {new_name}")
            self.invalidate_cache()
            return True
        except Error as e:
            logging.error(f"사용자 이름 업데이트 오류: {e}")
            self.connection.rollback()
            return False

    def get_all_users(self) -> List[Dict[str, Any]]:
        current_time = time.time()
        if current_time - self.last_cache_update > self.cache_ttl:
            try:
                with self.connection.cursor(dictionary=True) as cursor:
                    cursor.execute("SELECT id, name, embedding FROM users")
                    users = cursor.fetchall()
                    for user in users:
                        user['embedding'] = np.array(json.loads(user['embedding']))
                    self.users_cache = users
                    self.last_cache_update = current_time
            except Error as e:
                logging.error(f"사용자 정보 가져오기 오류: {e}")
                return []
        return self.users_cache

    def delete_user(self, user_id: int) -> bool:
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
            self.connection.commit()
            logging.info(f"사용자 삭제: {user_id}")
            self.invalidate_cache()
            return True
        except Error as e:
            logging.error(f"사용자 삭제 오류: {e}")
            self.connection.rollback()
            return False

    def check_face_similarity(self, new_embedding: np.ndarray) -> Tuple[bool, str]:
        try:
            users = self.get_all_users()
            for user in users:
                similarity = 1 - np.linalg.norm(new_embedding - user['embedding'])
                if similarity > 0.7:  # 유사도 임계값, 필요에 따라 조정 가능
                    return True, user['name']
            return False, ""
        except Exception as e:
            logging.error(f"얼굴 유사도 확인 오류: {e}")
            return False, ""

    def invalidate_cache(self):
        self.last_cache_update = 0

class FaceWorker(threading.Thread):
    def __init__(self, input_queue: Queue, output_queue: Queue, db_manager: DatabaseManager):
        threading.Thread.__init__(self)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.db_manager = db_manager
        self.running = True
        self.detection_backend = config.get('FaceRecognition', 'DetectionBackend', fallback='opencv')
        self.recognition_model = config.get('FaceRecognition', 'RecognitionModel', fallback='Facenet')
        self.distance_threshold = config.getfloat('FaceRecognition', 'DistanceThreshold', fallback=0.6)
        self.frame_skip = 2

    def run(self):
        frame_count = 0
        while self.running:
            try:
                frame = self.input_queue.get(timeout=1)
                frame_count += 1
                if frame_count % self.frame_skip == 0:
                    result = self.process_frame(frame)
                    self.output_queue.put(result)
            except Empty:
                continue
            except Exception as e:
                logging.error(f"FaceWorker 오류: {e}")

    def process_frame(self, frame):
        face_objs = DeepFace.extract_faces(frame, detector_backend=self.detection_backend, enforce_detection=False)
        results = []
        for face in face_objs:
            if isinstance(face, dict) and 'face' in face:
                face_img = face['face']
                embedding_obj = DeepFace.represent(face_img, model_name=self.recognition_model, enforce_detection=False)
                
                if isinstance(embedding_obj, list) and len(embedding_obj) > 0:
                    embedding = np.array(embedding_obj[0]['embedding'])
                elif isinstance(embedding_obj, dict) and 'embedding' in embedding_obj:
                    embedding = np.array(embedding_obj['embedding'])
                else:
                    continue
                
                name = self.recognize_face(embedding)
                results.append({
                    'name': name,
                    'bbox': face.get('facial_area', {}),
                    'embedding': embedding
                })
        return results

    def recognize_face(self, embedding: np.ndarray) -> str:
        min_distance = float('inf')
        recognized_name = "알 수 없음"
        for user in self.db_manager.get_all_users():
            try:
                distance = np.linalg.norm(embedding - user['embedding'])
                if distance < min_distance and distance < self.distance_threshold:
                    min_distance = distance
                    recognized_name = user['name']
            except Exception as e:
                logging.error(f"사용자 {user['name']}의 임베딩 비교 오류: {e}")
        return recognized_name

    def stop(self):
        self.running = False

class FaceApp:
    def __init__(self, db_manager, mode='recognition'):
        self.db_manager = db_manager
        self.mode = mode
        self.input_queue = Queue(maxsize=1)
        self.output_queue = Queue()
        self.worker = FaceWorker(self.input_queue, self.output_queue, self.db_manager)
        self.worker.start()
        self.cap = cv2.VideoCapture(0)
        self.running = True
        self.frame_buffer = None
        self.result_buffer = None
        self.font = ImageFont.truetype("C:/Windows/Fonts/gulim.ttc", 20)
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)
        self.last_greeting_time = {}
        self.greeting_cooldown = 60
        self.unknown_greeting_cooldown = 60
        self.last_unknown_greeting_time = 0
        self.current_unknown_id = 0

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                logging.error("프레임 캡처 실패")
                break

            frame = cv2.flip(frame, 1)

            if self.input_queue.empty():
                self.input_queue.put(frame)

            try:
                results = self.output_queue.get_nowait()
                self.result_buffer = results
                
                if self.mode == 'recognition':
                    self.handle_recognition_results()
                elif self.mode == 'registration':
                    self.handle_registration_results()

            except Empty:
                pass

            self.frame_buffer = frame.copy()
            self.display_results()

            cv2.imshow('얼굴 인식 시스템', self.frame_buffer)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r') and self.mode == 'registration':
                self.register_new_face()

        self.cleanup()

    def handle_recognition_results(self):
        current_time = time.time()
        recognized_names = set()
        unknown_detected = False

        for result in self.result_buffer:
            name = result['name']
            if name != "알 수 없음":
                recognized_names.add(name)
                if current_time - self.last_greeting_time.get(name, 0) >= self.greeting_cooldown:
                    self.greet_user(name)
                    self.last_greeting_time[name] = current_time
            else:
                unknown_detected = True
        
        if unknown_detected and current_time - self.last_unknown_greeting_time >= self.unknown_greeting_cooldown:
            self.greet_unknown_user()
            self.last_unknown_greeting_time = current_time
            self.current_unknown_id += 1

    def handle_registration_results(self):
        if self.result_buffer:
            result = self.result_buffer[0]  # 첫 번째 얼굴만 처리
            self.current_embedding = result['embedding']
            is_similar, similar_name = self.db_manager.check_face_similarity(self.current_embedding)
            if is_similar:
                self.similar_face = similar_name
            else:
                self.similar_face = None

    def display_results(self):
        if self.frame_buffer is not None and self.result_buffer is not None:
            pil_image = Image.fromarray(cv2.cvtColor(self.frame_buffer, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)
            
            for result in self.result_buffer:
                bbox = result['bbox']
                if isinstance(bbox, dict) and all(k in bbox for k in ('x', 'y', 'w', 'h')):
                    draw.rectangle([bbox['x'], bbox['y'], bbox['x']+bbox['w'], bbox['y']+bbox['h']], outline=(0, 255, 0), width=2)
                    name = result['name']
                    if name == "알 수 없음" and self.mode == 'recognition':
                        name = f"알 수 없음 {self.current_unknown_id}"
                    draw.text((bbox['x'], bbox['y']-20), f"이름: {name}", font=self.font, fill=(0, 255, 0))
            
            if self.mode == 'registration':
                if self.similar_face:
                    draw.text((10, 30), f"유사한 사용자: {self.similar_face}", font=self.font, fill=(255, 0, 0))
                    draw.text((10, 60), "다른 사람입니까? 'r' 키를 눌러 등록하세요", font=self.font, fill=(0, 255, 0))
                else:
                    draw.text((10, 30), "새로운 얼굴 등록: 'r' 키 누르기", font=self.font, fill=(0, 255, 0))
            
            self.frame_buffer = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def greet_user(self, name):
        message = f"{name}님, 안녕하세요."
        print(message)
        self.tts_engine.say(message)
        self.tts_engine.runAndWait()

    def greet_unknown_user(self):
        message = "환영합니다. 바이오브레인입니다."
        print(message)
        self.tts_engine.say(message)
        self.tts_engine.runAndWait()

    def register_new_face(self):
        if self.similar_face:
            confirm = input(f"유사한 사용자 {self.similar_face}가 감지되었습니다. 그래도 등록하시겠습니까? (y/n): ")
            if confirm.lower() != 'y':
                print("등록이 취소되었습니다.")
                return

        if self.current_embedding is None:
            print("등록할 얼굴이 감지되지 않았습니다. 다시 시도해주세요.")
            return

        name = input("등록할 이름을 입력하세요: ")
        if self.db_manager.user_exists(name):
            print(f"{name}은(는) 이미 등록된 이름입니다.")
            return

        success = self.db_manager.register_new_face(name, self.current_embedding)
        if success:
            print(f"{name} 등록 성공")
        else:
            print("등록 실패")

        self.current_embedding = None
        self.similar_face = None

    def cleanup(self):
        self.worker.stop()
        self.worker.join()
        self.cap.release()
        cv2.destroyAllWindows()

    def register_from_image(self, image_path: str):
        try:
            name = Path(image_path).stem
            if self.db_manager.user_exists(name):
                print(f"{name}은(는) 이미 등록되어 있습니다.")
                return

            image = cv2.imread(image_path)
            self.input_queue.put(image)
            
            try:
                results = self.output_queue.get(timeout=5)
                if results:
                    embedding = results[0]['embedding']
                    is_similar, similar_name = self.db_manager.check_face_similarity(embedding)
                    if is_similar:
                        print(f"{name}: 이미 등록된 얼굴과 유사합니다. 유사한 사용자: {similar_name}")
                        return
                    
                    success = self.db_manager.register_new_face(name, embedding)
                    if success:
                        print(f"{name} 등록 성공")
                    else:
                        print(f"{name} 등록 실패")
                else:
                    print(f"{name}: 얼굴이 감지되지 않았습니다")
            except Empty:
                print(f"{name}: 처리 시간 초과")
        except Exception as e:
            logging.error(f"이미지 등록 오류 ({image_path}): {e}")
            print(f"{name} 등록 중 오류 발생")

def display_users(db_manager):
    users = db_manager.get_all_users_info()
    if not users:
        print("등록된 사용자가 없습니다.")
        return
    
    print("\n등록된 사용자 목록:")
    print("ID\t이름\t\t등록일")
    print("-" * 40)
    for user in users:
        print(f"{user['id']}\t{user['name']}\t\t{user['created_at']}")

def update_user(db_manager):
    display_users(db_manager)
    user_id = input("\n수정할 사용자의 ID를 입력하세요: ")
    new_name = input("새로운 이름을 입력하세요: ")

    if db_manager.update_user_name(int(user_id), new_name):
        print("사용자 정보가 성공적으로 업데이트되었습니다.")
    else:
        print("사용자 정보 업데이트에 실패했습니다.")

def delete_user_interface(db_manager):
    display_users(db_manager)
    user_id = input("\n삭제할 사용자의 ID를 입력하세요: ")
    
    if db_manager.delete_user(int(user_id)):
        print("사용자가 성공적으로 삭제되었습니다.")
    else:
        print("사용자 삭제에 실패했습니다.")

def main():
    import io
    import sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')    
    db_manager = DatabaseManager()
    
    while True:
        print("\n얼굴 인식 시스템")
        print("1. 얼굴 인식 모드")
        print("2. 얼굴 등록 모드")
        print("3. 사용자 목록 조회")
        print("4. 사용자 정보 수정")
        print("5. 사용자 삭제")
        print("6. 종료")
        
        choice = input("선택하세요 (1-6): ")
        
        if choice == '1':
            app = FaceApp(db_manager, mode='recognition')
            app.run()
        elif choice == '2':
            app = FaceApp(db_manager, mode='registration')
            image_dir = config.get('Paths', 'ImageDirectory', fallback='')
            if image_dir and os.path.isdir(image_dir):
                for filename in os.listdir(image_dir):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_path = os.path.join(image_dir, filename)
                        app.register_from_image(image_path)
            app.run()
        elif choice == '3':
            display_users(db_manager)
        elif choice == '4':
            update_user(db_manager)
        elif choice == '5':
            delete_user_interface(db_manager)
        elif choice == '6':
            print("프로그램을 종료합니다.")
            break
        else:
            print("잘못된 선택입니다. 다시 시도해주세요.")

if __name__ == "__main__":
    main()