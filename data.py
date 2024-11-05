from PIL import Image
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

class ImageProcessor:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(-4, 4))

    def get_dataset(self, image_path, shuffle=True):
        img = Image.open(image_path).convert('L')
        width, height = img.size
        black_pixels = []
        for x in range(width):
            for y in range(height):
                pixel_value = img.getpixel((x, y))
                if pixel_value < 10:
                    black_pixels.append((x, y))
        
        rev_black_pixels = [(x, height - y) for x, y in black_pixels]
        left_rev_black_pixels = [(x, y) for x, y in rev_black_pixels if x < width / 2]
        right_rev_black_pixels = [(x, y) for x, y in rev_black_pixels if x >= width / 2]
        
        data_points = [(x, y, 0) for x, y in left_rev_black_pixels] + [(x, y, 1) for x, y in right_rev_black_pixels]
        
        data_points_array = np.array(data_points)
        X = data_points_array[:, :2]
        Y = data_points_array[:, 2]
        
        X_normalized = self.scaler.fit_transform(X)
        
        if shuffle:
            combined = np.column_stack((X_normalized, Y))
            np.random.shuffle(combined)
            
            # 分离X_shuffled和Y_shuffled
            X_shuffled = combined[:, :2]
            Y_shuffled = combined[:, 2]
        
            return X_shuffled, Y_shuffled
        else:
            return X_normalized, Y