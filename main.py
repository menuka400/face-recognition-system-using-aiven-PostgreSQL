import cv2
import numpy as np
from imgbeddings import imgbeddings
from PIL import Image
import psycopg2
import os
from tabulate import tabulate

def connect_db():
    return psycopg2.connect("postgres://avnadmin:AVNS_F9QTOJet-vzalfnV2-T@pg-9c3cfe6-menuka-1.h.aivencloud.com:21721/defaultdb?sslmode=require")

def initialize_database():
    """Create the required table if it doesn't exist"""
    conn = connect_db()
    cur = conn.cursor()
    try:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS face_records (
                id VARCHAR(255) PRIMARY KEY,
                embedding FLOAT[] NOT NULL
            )
        """)
        conn.commit()
        print("Database initialized successfully!")
    except Exception as e:
        print(f"Error initializing database: {str(e)}")
    finally:
        cur.close()
        conn.close()

def capture_face():
    cap = cv2.VideoCapture(0)
    alg = "haarcascade_frontalface_default.xml"
    haar_cascade = cv2.CascadeClassifier(alg)
    
    while True:
        ret, frame = cap.read()
        gray_img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = haar_cascade.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=2, minSize=(100, 100))
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        cv2.imshow("Face Detection - Press 's' to Save, 'q' to Quit", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s') and len(faces) > 0:
            x, y, w, h = faces[0]
            face = frame[y:y + h, x:x + w]
            cv2.imwrite("stored-faces/new_user.jpg", face)
            cap.release()
            cv2.destroyAllWindows()
            return "stored-faces/new_user.jpg"
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return None

def calculate_similarity(embedding1, embedding2):
    emb1 = np.array(embedding1)
    emb2 = np.array(embedding2)
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return similarity

def register_face():
    user_id = input("Enter ID number: ")
    
    conn = connect_db()
    cur = conn.cursor()
    try:
        # Check if user_id already exists
        cur.execute("SELECT id FROM face_records WHERE id = %s", (user_id,))
        if cur.fetchone():
            print("Error: This ID is already registered!")
            return
        
        img_path = capture_face()
        if img_path:
            img = Image.open(img_path)
            ibed = imgbeddings()
            embedding = ibed.to_embeddings(img)
            
            cur.execute("INSERT INTO face_records (id, embedding) VALUES (%s, %s)",
                       (user_id, embedding[0].tolist()))
            conn.commit()
            print(f"New user registered successfully with ID: {user_id}!")
            
    except Exception as e:
        print(f"Error during registration: {str(e)}")
    finally:
        cur.close()
        conn.close()

def recognize_face():
    img_path = capture_face()
    if img_path:
        try:
            img = Image.open(img_path)
            ibed = imgbeddings()
            current_embedding = ibed.to_embeddings(img)
            
            conn = connect_db()
            cur = conn.cursor()
            
            # Get all faces from database
            cur.execute("SELECT id, embedding FROM face_records")
            rows = cur.fetchall()
            
            if not rows:
                print("No registered users in the database!")
                return
            
            # Find the best match
            best_similarity = -1
            best_match_id = None
            similarity_threshold = 0.85
            
            for row in rows:
                user_id, stored_embedding = row
                similarity = calculate_similarity(current_embedding[0], stored_embedding)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_id = user_id
            
            if best_similarity >= similarity_threshold:
                print(f"User recognized as ID: {best_match_id}")
                print(f"Confidence: {best_similarity:.2%}")
            else:
                print("Access denied: Face not recognized")
                print(f"Best match confidence: {best_similarity:.2%}")
        
        except Exception as e:
            print(f"Error during recognition: {str(e)}")
        finally:
            cur.close()
            conn.close()

def view_and_edit_database():
    while True:
        print("\nDatabase Management Menu:")
        print("1. View all registered users")
        print("2. Remove a user")
        print("3. Return to main menu")
        
        choice = input("Enter your choice: ")
        
        if choice == "1":
            view_users()
        elif choice == "2":
            remove_user()
        elif choice == "3":
            break
        else:
            print("Invalid choice. Please try again.")

def view_users():
    conn = connect_db()
    cur = conn.cursor()
    try:
        cur.execute("SELECT id FROM face_records ORDER BY id")
        rows = cur.fetchall()
        
        if not rows:
            print("\nNo users registered in the database.")
            return
        
        # Prepare data for tabulate
        table_data = [[i+1, row[0]] for i, row in enumerate(rows)]
        headers = ["No.", "User ID"]
        
        print("\nRegistered Users:")
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        print(f"\nTotal users registered: {len(rows)}")
        
    except Exception as e:
        print(f"Error viewing users: {str(e)}")
    finally:
        cur.close()
        conn.close()

def remove_user():
    view_users()  # Show current users first
    
    user_id = input("\nEnter the ID of the user to remove (or press Enter to cancel): ")
    if not user_id:
        return
    
    conn = connect_db()
    cur = conn.cursor()
    try:
        # Check if user exists
        cur.execute("SELECT id FROM face_records WHERE id = %s", (user_id,))
        if not cur.fetchone():
            print(f"No user found with ID: {user_id}")
            return
        
        # Confirm deletion
        confirm = input(f"Are you sure you want to remove user {user_id}? (yes/no): ")
        if confirm.lower() != 'yes':
            print("Deletion cancelled.")
            return
        
        cur.execute("DELETE FROM face_records WHERE id = %s", (user_id,))
        conn.commit()
        print(f"User {user_id} has been removed successfully!")
        
    except Exception as e:
        print(f"Error removing user: {str(e)}")
    finally:
        cur.close()
        conn.close()

def main():
    os.makedirs("stored-faces", exist_ok=True)
    initialize_database()
    
    while True:
        print("\nMain Menu:")
        print("1. Start Face Recognition")
        print("2. Create New User")
        print("3. View User Database and Edit")
        print("4. Exit")
        
        choice = input("Enter your choice: ")
        
        if choice == "1":
            recognize_face()
        elif choice == "2":
            register_face()
        elif choice == "3":
            view_and_edit_database()
        elif choice == "4":
            print("Exiting program...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()