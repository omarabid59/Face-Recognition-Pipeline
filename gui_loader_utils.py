import glob
import numpy as np
import cv2
from PIL import Image
# Load the 3D Models
def load_3d_model_images(rootDir, width=400,height=400):
    '''
        This file loads the 3D image models that were extracted from the 3D Warehouse
        and converted into images from different angles using MeshLab on Ubuntu
        Returns: A dictionary with np array of all the files loaded
    '''

    pistol_model_files = sorted(glob.glob(rootDir + 'pistol/*.png'))
    machine_gun_model_files = sorted(glob.glob(rootDir + 'machine_gun/*.png'))
    mobile_model_files = sorted(glob.glob(rootDir + 'mobile/*.png'))
    default_model_files = sorted(glob.glob(rootDir + 'default/*.png'))
    car_grab_model_files = sorted(glob.glob(rootDir + 'number-plate/*.png'))
    vest_model_files = sorted(glob.glob(rootDir + 'vest/*.png'))
    pistol_models = []
    machine_gun_models = []
    mobile_models = []
    default_models = []
    car_grab_models = []
    vest_models = []
    model_3d_dict = dict()
    for file in pistol_model_files:
        tmp_img = np.array(Image.open(file))
        tmp_img = tmp_img[200:-300,300:-300] # Resize it to remove extra white space
        tmp_img = cv2.resize(tmp_img.copy(), dsize=(width,height), interpolation=cv2.INTER_CUBIC)
        pistol_models.append(tmp_img)
    for file in machine_gun_model_files:
        tmp_img = np.array(Image.open(file))
        tmp_img = cv2.resize(tmp_img.copy(), dsize=(width,height), interpolation=cv2.INTER_CUBIC)
        machine_gun_models.append(tmp_img)
    for file in mobile_model_files:
        tmp_img = np.array(Image.open(file))
        tmp_img = cv2.resize(tmp_img.copy(), dsize=(width,height), interpolation=cv2.INTER_CUBIC)
        mobile_models.append(tmp_img)
    for file in default_model_files:
        tmp_img = np.array(Image.open(file))
        tmp_img = cv2.resize(tmp_img.copy(), dsize=(width,height), interpolation=cv2.INTER_CUBIC)
        default_models.append(tmp_img)
    for file in car_grab_model_files:
        tmp_img = np.array(Image.open(file))
        tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB)
        tmp_img = cv2.resize(tmp_img.copy(), dsize=(width,height), interpolation=cv2.INTER_CUBIC)
        car_grab_models.append(tmp_img)
    for file in vest_model_files:
        tmp_img = np.array(Image.open(file))
        tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB)
        tmp_img = cv2.resize(tmp_img.copy(), dsize=(width,height), interpolation=cv2.INTER_CUBIC)
        vest_models.append(tmp_img)
    model_3d_dict['pistol'] = pistol_models
    model_3d_dict['machine_gun'] = machine_gun_models
    model_3d_dict['mobile'] = mobile_models
    model_3d_dict['default'] = default_models
    model_3d_dict['number-plate'] = car_grab_models
    model_3d_dict['vest'] = vest_models
    return model_3d_dict


def load_face_model_images(rootDir, width=400,height=400):
    '''
        This file loads the face image models that were extracted from the 3D Warehouse
        and converted into images from different angles using MeshLab on Ubuntu
        Returns: A dictionary with np array of all the files loaded
    '''
    model_files = sorted(glob.glob(rootDir + '*.jpg'))
    return load_images(model_files,width,height)

def load_side_view_gun_model_images(rootDir, width=400,height=400):
    '''
        This file loads the face image models that were extracted from the 3D Warehouse
        and converted into images from different angles using MeshLab on Ubuntu
        Returns: A dictionary with np array of all the files loaded
    '''
    model_files = sorted(glob.glob(rootDir + '*.jpg'))
    return load_images(model_files,width,height)

def load_images(model_files, width,height):
    '''
    Helper function to load the images for the side view.
    '''
    image_dict = dict()
    for file in model_files:
        tmp_img = np.array(Image.open(file))
        tmp_img = cv2.resize(tmp_img.copy(), dsize=(width,height), interpolation=cv2.INTER_CUBIC)
        tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB)
        file_name = file.split('/')[-1]
        file_name = file_name.split('.')[0]
        image_dict[file_name] = tmp_img
    return image_dict
