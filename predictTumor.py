from tensorflow.keras.models import load_model
import cv2 as cv
import numpy as np
import imutils

# Load the model
model = load_model('brain_tumor_detector.h5')

def predictTumor(image, use_ensemble=False):
    """
    Predicts whether an MRI image contains a tumor
    Returns: probability (float) between 0 and 1
    
    Parameters:
    - image: Input MRI image
    - use_ensemble: If True, uses multiple preprocessing methods for better accuracy
    """
    
    # Method 1: Standard preprocessing (original)
    def standard_preprocess(img):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        blurred = cv.GaussianBlur(gray, (5, 5), 0)
        equalized = cv.equalizeHist(blurred)
        processed = cv.cvtColor(equalized, cv.COLOR_GRAY2BGR)
        return processed
    
    # Method 2: CLAHE preprocessing (better contrast)
    def clahe_preprocess(img):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        blurred = cv.GaussianBlur(enhanced, (3, 3), 0)
        processed = cv.cvtColor(blurred, cv.COLOR_GRAY2BGR)
        return processed
    
    # Method 3: Sharpened preprocessing
    def sharpened_preprocess(img):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        blurred = cv.GaussianBlur(gray, (5, 5), 0)
        equalized = cv.equalizeHist(blurred)
        # Apply sharpening kernel
        kernel = np.array([[-1,-1,-1],
                           [-1, 9,-1],
                           [-1,-1,-1]])
        sharpened = cv.filter2D(equalized, -1, kernel)
        processed = cv.cvtColor(sharpened, cv.COLOR_GRAY2BGR)
        return processed
    
    # Method 4: Multi-scale preprocessing
    def multiscale_preprocess(img):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Apply multiple blur scales
        blur1 = cv.GaussianBlur(gray, (3, 3), 0)
        blur2 = cv.GaussianBlur(gray, (7, 7), 0)
        # Combine scales
        combined = cv.addWeighted(blur1, 0.6, blur2, 0.4, 0)
        equalized = cv.equalizeHist(combined)
        processed = cv.cvtColor(equalized, cv.COLOR_GRAY2BGR)
        return processed
    
    # Get model input shape
    expected_shape = model.input_shape
    if len(expected_shape) == 4:
        height, width = expected_shape[1], expected_shape[2]
    else:
        height, width = 150, 150
    
    if use_ensemble:
        # Use multiple preprocessing methods and average predictions
        methods = [standard_preprocess, clahe_preprocess, sharpened_preprocess, multiscale_preprocess]
        predictions = []
        
        for method in methods:
            processed = method(image)
            resized = cv.resize(processed, (width, height))
            resized = cv.cvtColor(resized, cv.COLOR_BGR2RGB)
            normalized = resized / 255.0
            input_data = np.expand_dims(normalized, axis=0)
            pred = model.predict(input_data, verbose=0)
            
            if isinstance(pred, (list, np.ndarray)):
                if pred.ndim == 2:
                    predictions.append(float(pred[0][0]))
                else:
                    predictions.append(float(pred[0]))
            else:
                predictions.append(float(pred))
        
        # Average predictions and add confidence score
        prob = np.mean(predictions)
        confidence_std = np.std(predictions)
        
        # If predictions vary widely, reduce confidence
        if confidence_std > 0.15:
            prob = prob * 0.8  # Reduce confidence for inconsistent predictions
        
        print(f"Ensemble prediction: {prob:.4f} (std: {confidence_std:.3f})")
        
    else:
        # Use adaptive preprocessing based on image characteristics
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        
        # Analyze image characteristics
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        print(f"Image analysis - Brightness: {brightness:.1f}, Contrast: {contrast:.1f}")
        
        # Adaptive preprocessing based on image quality
        if contrast < 40:  # Low contrast image
            print("Low contrast detected - Applying CLAHE")
            clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            processed = cv.cvtColor(enhanced, cv.COLOR_GRAY2BGR)
        elif brightness < 100:  # Dark image
            print("Dark image detected - Applying brightness enhancement")
            enhanced = cv.equalizeHist(gray)
            processed = cv.cvtColor(enhanced, cv.COLOR_GRAY2BGR)
        else:  # Normal image
            print("Normal image - Standard preprocessing")
            blurred = cv.GaussianBlur(gray, (5, 5), 0)
            equalized = cv.equalizeHist(blurred)
            processed = cv.cvtColor(equalized, cv.COLOR_GRAY2BGR)
        
        # Resize and predict
        resized = cv.resize(processed, (width, height))
        resized = cv.cvtColor(resized, cv.COLOR_BGR2RGB)
        
        # Apply normalization
        normalized = resized / 255.0
        
        # Optional: Apply data augmentation for better prediction
        # Flip horizontally
        flipped = np.fliplr(normalized)
        input_data1 = np.expand_dims(normalized, axis=0)
        input_data2 = np.expand_dims(flipped, axis=0)
        
        # Get predictions from both original and flipped
        pred1 = model.predict(input_data1, verbose=0)
        pred2 = model.predict(input_data2, verbose=0)
        
        # Average predictions
        if isinstance(pred1, (list, np.ndarray)):
            prob1 = float(pred1[0][0]) if pred1.ndim == 2 else float(pred1[0])
            prob2 = float(pred2[0][0]) if pred2.ndim == 2 else float(pred2[0])
        else:
            prob1 = float(pred1)
            prob2 = float(pred2)
        
        prob = (prob1 + prob2) / 2
        
        print(f"Predictions - Original: {prob1:.4f}, Flipped: {prob2:.4f}, Average: {prob:.4f}")
    
    # Apply final calibration based on empirical observations
    # Adjust these values based on your testing
    if prob > 0.95:
        prob = min(0.98, prob)  # Cap at 0.98
    elif prob < 0.1:
        prob = max(0.02, prob)  # Floor at 0.02
    
    # Apply sigmoid-like adjustment for better separation
    prob = 1 / (1 + np.exp(-5 * (prob - 0.5)))
    
    print(f"Final calibrated probability: {prob:.4f}")
    print("-" * 40)
    
    return prob

def predictTumorBatch(images):
    """
    Predict for multiple images at once
    Returns: list of probabilities
    """
    results = []
    for img in images:
        prob = predictTumor(img, use_ensemble=False)
        results.append(prob)
    return results

def getPredictionConfidence(probability):
    """
    Get confidence level and recommendation based on probability
    """
    if probability >= 0.85:
        return "HIGH", "Immediate medical consultation recommended"
    elif probability >= 0.70:
        return "MEDIUM", "Further testing recommended"
    elif probability >= 0.50:
        return "LOW", "Monitor and repeat scan if symptoms persist"
    else:
        return "VERY_LOW", "No immediate concern, regular checkup advised"

def cropTumorRegion(image):
    """
    Crops the tumor region from the image using contour detection
    Returns: cropped image
    """
    # Convert to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive thresholding for better contour detection
    thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv.THRESH_BINARY, 11, 2)
    
    # Morphological operations
    kernel = np.ones((5,5), np.uint8)
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
    thresh = cv.erode(thresh, kernel, iterations=1)
    thresh = cv.dilate(thresh, kernel, iterations=2)
    
    # Find contours
    cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    if not cnts:
        return image
        
    # Filter contours by area
    min_area = image.shape[0] * image.shape[1] * 0.01  # 1% of image area
    valid_cnts = [c for c in cnts if cv.contourArea(c) > min_area]
    
    if not valid_cnts:
        return image
        
    c = max(valid_cnts, key=cv.contourArea)
    
    # Get bounding rectangle
    x, y, w, h = cv.boundingRect(c)
    
    # Add padding (10%)
    padding_x = int(w * 0.1)
    padding_y = int(h * 0.1)
    
    x = max(0, x - padding_x)
    y = max(0, y - padding_y)
    w = min(image.shape[1] - x, w + 2*padding_x)
    h = min(image.shape[0] - y, h + 2*padding_y)
    
    # Crop image
    new_image = image[y:y+h, x:x+w]
    
    return new_image

# Example usage for testing
if __name__ == "__main__":
    # Test with both tumor and non-tumor images
    test_images = ['tumor.jpg', 'notumor.jpg']
    
    for img_path in test_images:
        print(f"\nTesting: {img_path}")
        test_image = cv.imread(img_path)
        if test_image is not None:
            probability = predictTumor(test_image, use_ensemble=True)
            confidence, recommendation = getPredictionConfidence(probability)
            print(f"Probability: {probability:.4f}")
            print(f"Confidence: {confidence}")
            print(f"Recommendation: {recommendation}")
            print("=" * 50)