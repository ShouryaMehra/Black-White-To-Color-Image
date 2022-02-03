import time
import numpy as np
import numpy
from collections import defaultdict
from scipy import stats
import cv2
from PIL import Image, ImageDraw
from flask import Flask,jsonify,request,send_file
import json
import io
from io import BytesIO
import os
from dotenv import load_dotenv

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
# set env for secret key
load_dotenv()

secret_id = os.getenv('AI_SERVICE_SECRET_KEY')

# print(secret_id)
def check_for_secret_id(request_data):    
    try:
        if 'secret_id' not in request_data.keys():
            return False, "Secret Key Not Found."
        
        else:
            if request_data['secret_id'] == secret_id:
                return True, "Secret Key Matched"
            else:
                return False, "Secret Key Does Not Match. Incorrect Key."
    except Exception as e:
        message = "Error while checking secret id: " + str(e)
        return False,message


@app.route('/Image_transformer',methods=['POST'])  #main function
def main():
    key = request.form['secret_id']
    request_data = {'secret_id' : key}
    secret_id_status,secret_id_message = check_for_secret_id(request_data)
    print ("Secret ID Check: ", secret_id_status,secret_id_message)
    if not secret_id_status:
        return jsonify({'message':"Secret Key Does Not Match. Incorrect Key.",
                        'success':False}) 
    else:
        img_params =request.files['image'].read()
        npimg = np.fromstring(img_params, np.uint8)
        #load image
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        pts = np.load('models/pts_in_hull.npy')
        net = cv2.dnn.readNetFromCaffe('models/colorization_deploy_v2.prototxt','models/colorization_release_v2.caffemodel')

        # add the cluster centers as 1x1 convolutions to the model
        class8 = net.getLayerId("class8_ab")
        conv8 = net.getLayerId("conv8_313_rh")
        pts = pts.transpose().reshape(2, 313, 1, 1)
        net.getLayer(class8).blobs = [pts.astype("float32")]
        net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
        # load the input image from disk, scale the pixel intensities to the
        # range [0, 1], and then convert the image from the BGR to Lab color
        # space
        #image = cv2.imread("img.jpeg")
        scaled = image.astype("float32") / 255.0
        lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

        # resize the Lab image to 224x224 (the dimensions the colorization
        # network accepts), split channels, extract the 'L' channel, and then
        # perform mean centering
        resized = cv2.resize(lab, (224, 224))
        L = cv2.split(resized)[0]
        L -= 50

        # pass the L channel through the network which will *predict* the 'a'
        # and 'b' channel values
        'print("[INFO] colorizing image...")'
        net.setInput(cv2.dnn.blobFromImage(L))
        ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

        # resize the predicted 'ab' volume to the same dimensions as our
        # input image
        ab = cv2.resize(ab, (image.shape[1], image.shape[0]))
        
        # grab the 'L' channel from the *original* input image (not the
        # resized one) and concatenate the original 'L' channel with the
        # predicted 'ab' channels
        L = cv2.split(lab)[0]
        colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
        
        # convert the output image from the Lab color space to RGB, then
        # clip any values that fall outside the range [0, 1]
        colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
        colorized = np.clip(colorized, 0, 1)
        
        # the current colorized image is represented as a floating point
        # data type in the range [0, 1] -- let's convert to an unsigned
        # 8-bit integer representation in the range [0, 255]
        colorized = (255 * colorized).astype("uint8")

        I = cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(I.astype('uint8'))
        file_object = io.BytesIO()
        img.save(file_object, 'PNG')
        file_object.seek(0)

        output = send_file(file_object, mimetype='image/PNG')  
    return output

if __name__ == '__main__':
    app.run()
