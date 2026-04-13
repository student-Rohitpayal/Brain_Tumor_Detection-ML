import cv2 as cv
import numpy as np
import imutils

class DisplayTumor:
    def __init__(self):
        self.image = None
        self.cv_image = None  # Add this attribute
        self.processed_image = None
        
    def readImage(self, image):
        self.image = image
        # Convert PIL image to OpenCV format
        if hasattr(image, 'width'):  # It's a PIL image
            import numpy as np
            self.cv_image = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)
        else:
            self.cv_image = image
            
    def removeNoise(self):
        if self.cv_image is None:
            return None
        # Apply noise removal
        kernel = np.ones((5,5), np.uint8)
        gray = cv.cvtColor(self.cv_image, cv.COLOR_BGR2GRAY)
        thresh = cv.threshold(gray, 45, 255, cv.THRESH_BINARY)[1]
        thresh = cv.erode(thresh, kernel, iterations=1)
        thresh = cv.dilate(thresh, kernel, iterations=1)
        return thresh
    
    def findTumorLocation(self):
        """Find and highlight tumor location"""
        if self.cv_image is None:
            return None
            
        result_image = self.cv_image.copy()
        
        # Convert to grayscale
        gray = cv.cvtColor(self.cv_image, cv.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv.GaussianBlur(gray, (5, 5), 0)
        
        # Apply threshold
        thresh = cv.threshold(blurred, 45, 255, cv.THRESH_BINARY)[1]
        
        # Apply morphological operations
        kernel = np.ones((5,5), np.uint8)
        thresh = cv.erode(thresh, kernel, iterations=2)
        thresh = cv.dilate(thresh, kernel, iterations=2)
        
        # Find contours
        cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        
        if cnts:
            # Find the largest contour
            largest_contour = max(cnts, key=cv.contourArea)
            area = cv.contourArea(largest_contour)
            
            if area > 100:
                # Get bounding box
                x, y, w, h = cv.boundingRect(largest_contour)
                
                # Draw rectangle
                cv.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 3)
                
                # Draw circle at center
                center_x = x + w//2
                center_y = y + h//2
                cv.circle(result_image, (center_x, center_y), 5, (0, 0, 255), -1)
                
                # Add text
                cv.putText(result_image, "Tumor Region", (x, y-10), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv.drawContours(result_image, [largest_contour], -1, (255, 0, 0), 2)
            else:
                cv.putText(result_image, "No Significant Tumor Detected", (50, 50), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv.putText(result_image, "No Tumor Detected", (50, 50), 
                      cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return result_image
    
    def displayTumor(self):
        """Display the tumor location"""
        if self.cv_image is not None:
            highlighted_image = self.findTumorLocation()
            if highlighted_image is not None:
                # Resize for display if too large
                height, width = highlighted_image.shape[:2]
                if width > 800:
                    scale = 800 / width
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    highlighted_image = cv.resize(highlighted_image, (new_width, new_height))
                
                cv.imshow('Tumor Detection Result', highlighted_image)
                cv.waitKey(0)
                cv.destroyAllWindows()
                return highlighted_image
        else:
            print("No image loaded for tumor display")
            return None