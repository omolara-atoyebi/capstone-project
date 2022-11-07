import streamlit as st
import pickle
import mediapipe as mp
import cv2
import numpy as np
import tempfile
from PIL import Image
import pandas as pd


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
picture = Image.open('./video/streamlitVid/my_pic.jpg')
workflow = './video/streamlitVid/workflow3.jpg'
DEMO_VIDEO = './video/streamlitVid/DEMOVID.mp4'
vid = './video/streamlitVid/app_demo.webm'

with open('swim_style.pkl','rb') as f:
    model = pickle.load(f)

st.title('Detecting Swimming Styles(freestyle, breaststroke and backstroke)')

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title('Detecting Swimming Styles using MediaPipe')
st.sidebar.subheader('Parameters')

@st.cache()
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized

app_mode = st.sidebar.selectbox('Choose the App mode',
['About Me','About App','Test the Model']
)

if app_mode == 'About Me':
    st.header('Profile')
    
    st.image(picture, caption= 'Atoyebi Omolara', width= 150)
    st.markdown("""
                Hey, this is Atoyebi omolara, a master graduate of research and analytical chemistry, 
                who has found interest in Data science and Analysis and proceeded to acquire a Certificate in it by 
                attending EPICODE.
                Thanks to EPICODE for the intense teaching and hands on experience as well as the knowledge and skills 
                i acquired through her teaching Methodology.

                If you are interested in my profile :\n
                - You can check my LinkedIn profile from [LinkedIn](https://www.linkedin.com/in/omolara-mustophat-atoyebi-056345236/)\n
                - You can check my GitHub profile from [GitHub](https://github.com/omolara-atoyebi)\n
                - Email me @ atoyebiomolara@gmail.com

                """)


elif app_mode == 'About App':
    st.image(workflow)
    st.text('Demo output for acquiring Landmarks')
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    </style>
    """,
    unsafe_allow_html=True,
    )
    dem = open(vid,'rb')
    out_vid = dem.read()
    st.video(out_vid)
elif app_mode =='Test the Model':

    st.subheader('Testing the model')
    st.set_option('deprecation.showfileUploaderEncoding', False)

    use_webcam = st.sidebar.button('Use Webcam')
    
    st.sidebar.text('Params For video')
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,
        )
    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value =0.0,max_value = 1.0,value = 0.5)
    tracking_confidence = st.sidebar.slider('Min Tracking Confidence', min_value = 0.0,max_value = 1.0,value = 0.5)
    #max person
    max_person = st.sidebar.number_input('Maximum Number of person',value =1,min_value= 1, max_value=1)

    st.markdown("""Video or image should be capture from the side , underwater or top  view. please do not capture from the upper(head) or lower(toe)""")
    st.markdown("""------""")
   
    stframe = st.empty()
    

    video_file_buffer = st.sidebar.file_uploader("Upload a video", type=[ "mp4", "mov",'avi','asf', 'm4v','jpeg','jpg','png' ])





    

    tfflie = tempfile.NamedTemporaryFile(delete=False)


    if not video_file_buffer:
        if use_webcam:
            vid = cv2.VideoCapture(0)
        else:
            vid = cv2.VideoCapture(DEMO_VIDEO)
            tfflie.name = DEMO_VIDEO

    
    else:
        tfflie.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tfflie.name)



    

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    #codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
    codec = cv2.VideoWriter_fourcc('V','P','0','9')
    out = cv2.VideoWriter('output1.webm', codec, fps, (width, height))


    st.sidebar.text('Input Video')
    st.sidebar.video(tfflie.name)



    with mp_pose.Pose(
            #enable_segmentation=True,
            min_detection_confidence = detection_confidence,
            min_tracking_confidence = tracking_confidence
            ) as pose:
        while vid.isOpened():
            ret,frame = vid.read()
            if not ret:
                break
            # If loading a video, use 'break' instead of 'continue'.
                
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable =False

            results = pose.process(image)
            #print(results)

            image.flags.writeable =True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    #extract landmarks 
            try:
                pose_lan = results.pose_landmarks.landmark
                pose_row = list(np.array([[landmark.x,landmark.y,landmark.z,landmark.visibility]for landmark in pose_lan]).flatten())
                row= pose_row
            
                
                # Make Detections
                X = pd.DataFrame([row])
                body_language_class = model.predict(X)[0]
                body_language_prob = model.predict_proba(X)[0]
                print(body_language_class, body_language_prob)
            
           
            
                # Get status box
                cv2.rectangle(image, (0,0), (350, 100), (245, 117, 16), -1)
                
                # Display Class
                cv2.putText(image, 'CLASS'
                            , (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(image, body_language_class.split(' ')[0]
                            , (95,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                

                # Display Probability
                # cv2.putText(image, 'PROB'
                #             , (15,20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
                # cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)],2))
                #             , (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            except:
                pass


            out.write(image)    
            image = cv2.resize(image,(0,0),fx = 0.8 , fy = 0.8)

            image = image_resize(image, width = 640)
            
            stframe.image(image,channels = 'BGR',use_column_width=True)

    

    st.text('Video Processed')

    output_video = open('output1.webm','rb')
    out_bytes = output_video.read()
    st.video(out_bytes)


        





    vid.release()
    out.release()
