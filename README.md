# 🚀 Face Recognition System

A powerful and efficient **Face Recognition System** built using OpenCV, imgbeddings, and PostgreSQL. This project enables real-time face detection, registration, and recognition with database storage for embeddings.

## ✨ Features
- 🔍 **Real-time Face Detection** using OpenCV
- 🏷 **User Registration** with automatic embedding storage
- 🤖 **Face Recognition** with high accuracy
- 📊 **Database Management** (View, Edit, and Delete Users)
- 🚀 **PostgreSQL Integration** for scalable storage

## 📸 How It Works
1. Capture a face using your webcam.
2. Generate an embedding using `imgbeddings`.
3. Store the embedding in a PostgreSQL database.
4. Match faces against the stored database for recognition.

## 🔍 what is the Logic behind this?
**1. Vector representation, embeddings and search.**

Information can be stored in several ways, just think about the sentence `I Love Parks`:
you could represent it in a table with three columns to flag the presence or not of each word (`I`, `LOVE` and `PARKS`) as per image below:

![image](https://github.com/user-attachments/assets/c57920ce-88d1-4ca8-9242-31347b45c131)


This is a lossless method, no information (apart from the order of words) is lost with this 
encoding. The drawback though is that the number of columns grows with the number of distinct words within the sentence. For example, if we try to also encode
`I Love Croissants` with the same structure we'll end up with four columns `I`, `LOVE`, `PARKS` and `CROISSANTS` as shown below.

![image](https://github.com/user-attachments/assets/83884019-2e78-4aa9-baab-475cffc9f8f1)

**2. Embeddings**

What are embeddings then? As mentioned above, storing the presence of each word in a 
separate column would create a very wide and unmanageable dataset. Therefore a standard 
approach is to try to reduce the dimensionality by aggregating or dropping some of the 
redundant or not very distiguishable information. In our previous example, we could still 
encode the same information by:

i. dropping the `I` column since it doesn't add any value (it's always 1)

ii. dropping the `CROISSANTS` column since we can still distinguish the two sentences by the presence of the `PARK` word.

If we visualize the two sentences above in a graph only using the `LOVE` and `PARKS` axis 
(therefore excluding the `I` and `CROISSANTS`), the result shows that `I Love Parks` is 
encoded as `(1,1)` since it has present both the `LOVE` and the `PARKS` words. On the other 
hand `I Love Croissants` is encoded with `(1,0)` since it includes `LOVE` but not `PARKS`.

![image](https://github.com/user-attachments/assets/187b2f5a-e969-4e99-bd1c-abd5fb1fd654)

In the graph above, the `distance` represents a calculation of similarity between two vectors: 
The more two vectors point to the same direction or are close to each other, the more the 
information they represent should be similar.

**3. Does this work with pictures?**

A similar approach also works for pictures. As beautifully explained by [Mathias Grønne](https://towardsdatascience.com/introduction-to-image-embedding-and-accuracy-53473e8965f/) and 
visualized in the image below (it's Figure 1.1 from the above blog, original book photo photo by 
[Jess Bailey](https://unsplash.com/photos/gL2jT6xHYOY) on [Unsplash](https://unsplash.com/)), an image is just a series of characters in a matrix, and therefore we 
could reduce the matrix information and create embeddings on it.

![image](https://github.com/user-attachments/assets/be353354-439e-42ff-929f-157d8bf782af)

**4. Setup Face recognition with Python and PostgreSQL `pgvector`**

If you, like me, use IPhotos on Mac, you’ll be familiar with the “People” tab, where you can 
select one person and find the photos where this person is included. I used the following 
code to do the same sort of thing with the pictures coming from Crab Week - you’re invited to 
run it, with adaptations, on top of any folder containing images.

Since images are sensitive data, we don't want to rely on any online service or upload them to 
the internet. The entire pipeline defined below is working 100% locally.

The data pipeline will involve several steps:

i. Download all the pictures in a local folder
ii. Retrieve the faces included in any picture
iii. Calculate the embeddings from the faces
iv. Store the embedidngs in PostgreSQL in a `vector` column from `pgvector`
v. Get a colleague picture from Slack
vi. Identify the face in the picture (needed since people can have all types of pictures in Slack)
vii. Calculate the embeddings in the Slack picture
viii. Use `pgvector` distance function to retrieve the closest faces and therefore photos

The entire flow is shown in the picture below:

![image](https://github.com/user-attachments/assets/ff75acc8-6be1-4020-98a1-3bfb788e66ac)

In the graph above, the `distance` represents a calculation of similarity between two vectors: 
The more two vectors point to the same direction or are close to each other, the more the 
information they represent should be similar.

**5. Retrieve the faces from photos**

An ideal dataset to calculate embeddings would contain only pictures of one person at a time, 
looking straight in the camera with minimal background. As we know, this is not the truth for 
event pictures, where a multitude of people is commonly grouped together with various 
backgrounds. Therefore, to create a machine learning algorithm that will be able to find a 
person included in a picture, we need to isolate the faces of the people within the photos and 
create the embeddings on the faces rather than over the entire photos.

![image](https://github.com/user-attachments/assets/db6e5279-bb5c-44c9-8201-f5de0a097048)

To "extract" faces from the pictures we used Python, OpenCV a computer vision tool and a 
pre-trained Haar Cascade model.

To get it working, we just need to install the `opencv-python` package with:
```bash
pip install opencv-python
```
Download the `haarcascade_frontalface_default.xml` pre-trained Haar Cascade model 
from the [OpenCV GitHub repository](https://github.com/opencv/opencv/tree/master/data/haarcascades) and store it locally.

Insert the code below in a python file, replacing the `<INSERT YOUR IMAGE NAME HERE>` 
with the path to the image you want to identify faces from and 
`<INSERT YOUR TARGET IMAGE NAME HERE>` to the name of the file where you want to 
store the face.

```bash
# import the OpenCV library - it's called cv2
import cv2
# load the Haar Cascade algorithm from the XML file into OpenCV
alg = "haarcascade_frontalface_default.xml"
haar_cascade = cv2.CascadeClassifier(alg)
# read the image as grayscale
file_name = '<INSERT YOUR IMAGE NAME HERE>'
img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
# find the faces in that image
# this gives back an array of face locations and sizes
faces = haar_cascade.detectMultiScale(
    gray_img,
    scaleFactor=1.05,
    minNeighbors=2,
    minSize=(100, 100)
)
# for each face detected
for x, y, w, h in faces:
    # crop the image to select only the face
    cropped_image = img[y : y + h, x : x + w]
    # write the cropped image to a file
    target_file_name = '<INSERT YOUR TARGET IMAGE NAME HERE>'
    cv2.imwrite(
        target_file_name,
        cropped_image,
    )
```
The line that performs the magic is:

```bash
faces = haar_cascade.detectMultiScale(
    gray_img,
    scaleFactor=1.05,
    minNeighbors=2,
    minSize=(100, 100)
)
```
Where:

i. `gray_img` is the source image in which we need to find faces
ii. `scaleFactor` is the scaling factor, the higher ratio the more compression and more loss in image quality
iii. `minNeighbors` the amount of neighbour faces to collect. The higher the more the same face could appear multiple times.
iv. `minSize` the minimum size of a detected face, in this case a square of 100 pixels.

The `for` loop iterates over all the faces detected and stores them in separate files; you 
might want to define a variable (maybe using the `x` and `y` parameters) to store the various 
faces in different files. Moreover, if you plan to calculate embeddings over a series of pictures, 
you'll want to encapsulate the above code in a loop parsing all the files in a specific folder.

The result of the face detection stage is not perfect: it identifies three faces out of the four 
that are visible, but is good enough for our purpose. You can fine tune the algorithm 
parameters to find the better fit for your use cases.

**6. Calculate the embeddings**

Once we identified the faces, we can now calculate the embeddings. For this step we are 
going to use [imgbeddings](https://github.com/minimaxir/imgbeddings), a Python package to generate embedding vectors from images, 
using [OpenAI](https://openai.com/)'s [CLIP model](https://github.com/openai/CLIP) via [Hugging Face](https://huggingface.co/) [transformers](https://huggingface.co/docs/transformers/index).

To calculate the embeddings of a picture, we need to first install the required packages via

```bash
pip install imgbeddings
pip install pillow 
```
And then include the following in a Python file

```bash
# import the required libraries
import numpy as np
from imgbeddings import imgbeddings
from PIL import Image
# load the face image from its file
file_name = "INSERT YOUR FACE FILE NAME"
img = Image.open(file_name)
# loading `imgbeddings` so we can calculate embeddings
ibed = imgbeddings()
# calculating the embedding for our image
embedding = ibed.to_embeddings(img)[0]
```
The code above calculates the embeddings. The result is a 768 element numpy vector for 
each input image, representing its embedding.

**7. Store embeddings in PostgreSQL using `pgvector`**

It's time to start using the capability of PostgreSQL and the `pgvector` extension. First of all 
we need a PostgreSQL up and running, we can navigate to the [Aiven Console](https://console.aiven.io/), create a new 
PostgreSQL selecting the favourite cloud provider, region and plan and enabling extra disk 
storage if needed. The `pgvector` extension is available in all plans. Once all the settings are 
ok, you can click on Create Service.

Once the service is up and running (it can take a couple of minutes), navigate to the service 
Overview and copy the Service URI parameter. We'll use it to connect to PostgreSQL via [psql](https://aiven.io/docs/products/postgresql/howto/connect-psql) 
with:

```bash
psql <SERVICE_URI>
```

Once connected, we can enable the pgvector extension with:

```bash
CREATE EXTENSION vector;
```

And now we can create a table containing the picture name, and the embeddings with:

```bash
CREATE TABLE pictures (picture text PRIMARY KEY, embedding vector(768));
```

Check out the `embedding vector(768)`, we are defining a vector of 768 dimensions, 
exactly the same dimension as the output of the `ibed.to_embeddings(img)` function in the previous step.

To load the embedding in postgreSQL we can use [psycopg2](https://aiven.io/docs/products/postgresql/howto/connect-python) by installing it with

```bash
pip install psycopg2
```

and then using the following Python code always replacing the `<SERVICE_URI>` with the 
service URI

```bash
# import the required libraries
import psycopg2
# connect to our database and upload the record
conn = psycopg2.connect('<SERVICE_URI>')
cur = conn.cursor()
cur.execute('INSERT INTO pictures values (%s,%s)', (file_name, embedding.tolist()))
conn.commit()
conn.close()
```

Where `file_name` and `embedding` are the variables from the previous Python statement.

**8. Get Slack image, retrieve face and calculate embeddings**

The following steps in the process are similar to the ones already done above, this time the 
source image is the Slack profile picture where we'll detect the face and calculate the 
embeddings. The code above can be reused by changing the location of the source image.

![image](https://github.com/user-attachments/assets/4889e82b-e6e2-42c3-af2e-1245b86d09db)

The code below can give you a starting point

```bash
# load the image you want to search with
file_name = '<INSERT YOUR SLACK IMAGE NAME HERE>'
img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
# find the faces
faces = haar_cascade.detectMultiScale(
    gray_img,
    scaleFactor=1.05,
    minNeighbors=2,
    minSize=(100, 100)
)
# load `imgbeddings` so we can calculate embeddings
ibed = imgbeddings()
# for each face detected in the Slack picture
for x, y, w, h in faces:
    # crop the image to select only the face
    cropped_image = img[y : y + h, x : x + w]
    # calculating its embedding
    slack_img_embedding = ibed.to_embeddings(cropped_image)[0]
```

**9. Find similar images with vector search**

The final piece of the puzzle is to use the similarity functions available in pgvector to find 
pictures where the person is included. pgvector provides [different similarity functions](https://aiven.io/docs/products/postgresql/concepts/pgvector#vector-similarity), 
depending on the type of search we are trying to perform.

We'll use the [distance](https://github.com/pgvector/pgvector#distances) function, that calculates the euclidean distance between two vectors 
for our search. To find the other pictures with closest distance we can use the following query in Python:

```bash
conn = psycopg2.connect('<SERVICE_URI>')
cur = conn.cursor()
string_representation = "".join(str(x) for x in slack_img_embedding.tolist())
cur.execute("SELECT picture FROM pictures ORDER BY embedding <-> %s LIMIT 5;", (string_rep,))
rows = cur.fetchall()
for row in rows:
    print(row)
```
Where `slack_img_embedding` is the embeddings vector calculated on top of the Slack 
profile picture at the previous step. If everything is working correctly, you'll be able to see the 
name of top 5 pictures that are similar to the Slack profile image as input.

The results, in the crabweek case was five photos where my colleague Tibs was included!

![image](https://github.com/user-attachments/assets/e872b032-c6bc-4a10-a515-f7ae478c5517)

**10. pgvector, enabling Machine Learning in PostgreSQL**

Machine Learning is becoming pervasive in our day to day activities. Being able to store, 
query and analyse data embeddings in the same technology where the data resides, like a 
PostgreSQL database, can provide a number of benefits in machine learning democratisation 
and enable new use cases achievable by a standard SQL query.

To know more about pgvector and Machine Learning in PostgreSQL:

there's a Jupyter notebook containing a worked example of the above code [over at](https://github.com/Aiven-Labs/pgvector-image-recognition) 
pgvector [use cases and features description](https://aiven.io/docs/products/postgresql/concepts/pgvector)
[How to enable pgvector in Aiven for PostgreSQL](https://aiven.io/docs/products/postgresql/howto/use-pgvector)
pgvector [README on GitHub](https://github.com/pgvector/pgvector)

## 🛠 Setup & Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/face-recognition-system.git
cd face-recognition-system
```
Since Slack pictures could be complex, the above code has a `for` loop iterating over all the 
detected faces. You might want to add additional checks to find the most relevant face to 
calculate the embeddings from.

### 2️⃣ Install Dependencies
Ensure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

### 3️⃣ Set Up PostgreSQL
Follow these steps to set up PostgreSQL:

#### 🏗 Step 1: Create a PostgreSQL Service on Aiven
1. **Create an account** on [Aiven.io](https://aiven.io).
2. **Create a new project** in Aiven.
3. **Create a PostgreSQL service** (you can select the free plan) and skip all other configurations.
4. **Wait for the server** to transition from *Rebuilding* to *Running*.

#### 🔧 Step 2: Install PostgreSQL Locally
1. Download & install **PostgreSQL** from [EnterpriseDB](https://www.enterprisedb.com/downloads/postgres-postgresql-downloads).
2. Restart your computer after installation (this is required).

#### 🌐 Step 3: Connect to Your PostgreSQL Service
1. Go to **Aiven.io** → **Services**.
2. Select the **PostgreSQL service** you created.
3. Click **Quick Connect** → Choose **PSQL** as the connection method.
4. Copy the provided connection command and **paste it into your terminal**.

#### 📜 Step 4: Initialize the Database
After connecting to PostgreSQL, run the following commands your terminal:

```sql
CREATE EXTENSION vector;

CREATE TABLE pictures (picture text PRIMARY KEY,embedding vector(768));
```
5. **Update the `<SERVICE_URI>`** in your `face_recognition.py` code with your Aiven service URI.
   
![image](https://github.com/user-attachments/assets/e83297d5-01b7-4a56-865f-d66b71408aed)


### 4️⃣ Run the Application
```bash
python face_recognition.py
```

## 🎯 Usage
- Select an option from the menu:
  1. **Start Face Recognition**
  2. **Register a New User**
  3. **Manage User Database**
- Follow on-screen instructions to register or recognize faces.

## 📌 Contributing
Want to improve this project? Feel free to fork and contribute! Open an issue for any suggestions.

## 📜 License
This project is licensed under the MIT License.

🚀 Happy Coding! 🎯

🌟 **Star this repository if you found it useful!** 🌟

