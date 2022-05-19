from tensorflow import keras
import numpy as np
from keras.preprocessing import image
if "__main__":
    
    model = keras.models.load_model('./facee_model.h5')

    
    image_path = "./a.jpg"
    test_image = image.load_img(image_path, target_size=(48, 48), color_mode="grayscale")
    test_data = image.img_to_array(test_image)
    test_data = np.expand_dims(test_data, axis=0)
    test_data = np.vstack([test_data])
    results = model.predict(test_data, batch_size=1)
    class_names = ['kizgin', 'igrenme', 'korku', 'mutlu', 'uzgun', 'sasirma', 'dogal']
    a = np.argmax(results)
    print(a)
    self.lbl_tahmin_yazi.setText("sınıflandırma sonucu en yüksek oranla"+class_names[np.argmax(results)])