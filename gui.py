import tkinter
from PIL import Image, ImageTk
from tkinter import filedialog, ttk
import cv2 as cv
import numpy as np
import os
from datetime import datetime
from frames import *
from displayTumor import *
from predictTumor import *

class Gui:
    MainWindow = 0
    listOfWinFrame = list()
    FirstFrame = object()
    val = 0
    fileName = 0
    DT = object()

    wHeight = 700
    wWidth = 1180

    def __init__(self):
        global MainWindow
        MainWindow = tkinter.Tk()
        MainWindow.title("Brain Tumor Detection System")
        MainWindow.geometry('1200x720')
        MainWindow.resizable(width=False, height=False)
        
        # Initialize DisplayTumor
        self.DT = DisplayTumor()
        
        # Configure styles
        self.configure_styles()
        
        self.fileName = tkinter.StringVar()
        self.prediction_history = []  # Store prediction history
        self.current_image_path = None
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create main frames
        self.create_main_frames()
        
        # Create status bar
        self.create_status_bar()
        
        MainWindow.mainloop()
    
    def configure_styles(self):
        """Configure custom styles for the application"""
        self.bg_color = "#f0f0f0"
        self.primary_color = "#2c3e50"
        self.success_color = "#27ae60"
        self.warning_color = "#f39c12"
        self.error_color = "#e74c3c"
        self.info_color = "#3498db"
        
        MainWindow.configure(bg=self.bg_color)
    
    def create_menu_bar(self):
        """Create menu bar with additional options"""
        menubar = tkinter.Menu(MainWindow)
        MainWindow.config(menu=menubar)
        
        # File menu
        file_menu = tkinter.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Image", command=self.browseWindow, accelerator="Ctrl+O")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=MainWindow.quit)
        
        # View menu
        view_menu = tkinter.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Show Original", command=self.show_original)
        view_menu.add_command(label="Show Processed", command=self.show_processed)
        
        # Help menu
        help_menu = tkinter.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="Instructions", command=self.show_instructions)
    
    def create_main_frames(self):
        """Create main application frames"""
        # Left panel for controls
        self.left_panel = tkinter.Frame(MainWindow, width=400, bg=self.bg_color, relief=tkinter.RAISED, bd=2)
        self.left_panel.pack(side=tkinter.LEFT, fill=tkinter.Y, padx=10, pady=10)
        
        # Right panel for image display
        self.right_panel = tkinter.Frame(MainWindow, bg=self.bg_color, relief=tkinter.SUNKEN, bd=2)
        self.right_panel.pack(side=tkinter.RIGHT, fill=tkinter.BOTH, expand=True, padx=10, pady=10)
        
        # Create title in left panel
        title_label = tkinter.Label(self.left_panel, text="🧠 BRAIN TUMOR DETECTION", 
                                     font=("Arial", 18, "bold"), bg=self.bg_color, fg=self.primary_color)
        title_label.pack(pady=20)
        
        # Subtitle
        subtitle_label = tkinter.Label(self.left_panel, text="Medical Image Analysis System", 
                                        font=("Arial", 10), bg=self.bg_color, fg="#7f8c8d")
        subtitle_label.pack(pady=(0, 30))
        
        # Create info frame
        self.info_frame = tkinter.LabelFrame(self.left_panel, text="Image Information", 
                                              font=("Arial", 10, "bold"), bg=self.bg_color)
        self.info_frame.pack(fill=tkinter.X, padx=20, pady=10)
        
        self.image_info_label = tkinter.Label(self.info_frame, text="No image loaded", 
                                               font=("Arial", 9), bg=self.bg_color, justify=tkinter.LEFT)
        self.image_info_label.pack(padx=10, pady=10)
        
        # Create selection frame
        selection_frame = tkinter.LabelFrame(self.left_panel, text="Analysis Options", 
                                              font=("Arial", 10, "bold"), bg=self.bg_color)
        selection_frame.pack(fill=tkinter.X, padx=20, pady=10)
        
        # Radio buttons
        self.val = tkinter.IntVar(value=1)
        
        self.rb1 = tkinter.Radiobutton(selection_frame, text="🔍 Detect Tumor", variable=self.val,
                                       value=1, command=self.check, font=("Arial", 11), 
                                       bg=self.bg_color, selectcolor=self.bg_color)
        self.rb1.pack(anchor=tkinter.W, padx=20, pady=10)
        
        self.rb2 = tkinter.Radiobutton(selection_frame, text="📍 View Tumor Region", variable=self.val,
                                       value=2, command=self.check, font=("Arial", 11),
                                       bg=self.bg_color, selectcolor=self.bg_color)
        self.rb2.pack(anchor=tkinter.W, padx=20, pady=10)
        
        self.rb3 = tkinter.Radiobutton(selection_frame, text="📊 Full Analysis", variable=self.val,
                                       value=3, command=self.check, font=("Arial", 11),
                                       bg=self.bg_color, selectcolor=self.bg_color)
        self.rb3.pack(anchor=tkinter.W, padx=20, pady=10)
        
        # Button frame
        button_frame = tkinter.Frame(self.left_panel, bg=self.bg_color)
        button_frame.pack(pady=20)
        
        # Browse button
        self.browseBtn = tkinter.Button(button_frame, text="📁 Browse Image", width=15, 
                                        command=self.browseWindow, font=("Arial", 10, "bold"),
                                        bg=self.info_color, fg="white", relief=tkinter.RAISED)
        self.browseBtn.pack(side=tkinter.LEFT, padx=5)
        
        # View button
        self.viewBtn = tkinter.Button(button_frame, text="👁️ View", width=10, 
                                      command=self.view_image, font=("Arial", 10, "bold"),
                                      bg=self.primary_color, fg="white", state=tkinter.DISABLED)
        self.viewBtn.pack(side=tkinter.LEFT, padx=5)
        
        # Clear button
        self.clearBtn = tkinter.Button(button_frame, text="🗑️ Clear", width=10, 
                                       command=self.clear_all, font=("Arial", 10, "bold"),
                                       bg=self.warning_color, fg="white")
        self.clearBtn.pack(side=tkinter.LEFT, padx=5)
        
        # Results frame
        self.results_frame = tkinter.LabelFrame(self.left_panel, text="Analysis Results", 
                                                 font=("Arial", 10, "bold"), bg=self.bg_color)
        self.results_frame.pack(fill=tkinter.BOTH, expand=True, padx=20, pady=10)
        
        self.result_label = tkinter.Label(self.results_frame, text="No analysis performed yet", 
                                           font=("Arial", 11), bg=self.bg_color, wraplength=350)
        self.result_label.pack(padx=10, pady=20)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.left_panel, mode='indeterminate', length=300)
        self.progress.pack(pady=10)
        
        # Image display area
        self.image_display_label = tkinter.Label(self.right_panel, bg="white", relief=tkinter.SUNKEN, bd=2)
        self.image_display_label.pack(fill=tkinter.BOTH, expand=True, padx=10, pady=10)
        self.image_display_label.config(text="No Image Loaded\n\nClick 'Browse Image' to select an MRI scan", 
                                        font=("Arial", 12), fg="gray")
    
    def create_status_bar(self):
        """Create status bar at bottom"""
        self.status_bar = tkinter.Label(MainWindow, text="Ready", bd=1, relief=tkinter.SUNKEN, 
                                         anchor=tkinter.W, font=("Arial", 9))
        self.status_bar.pack(side=tkinter.BOTTOM, fill=tkinter.X)
    
    def update_status(self, message):
        """Update status bar message"""
        self.status_bar.config(text=message)
        MainWindow.update_idletasks()
    
    def browseWindow(self):
        """Open file dialog to browse images"""
        FILEOPENOPTIONS = dict(defaultextension='*.*',
                               filetypes=[('Image Files', '*.jpg *.jpeg *.png *.bmp *.tiff'),
                                         ('JPEG', '*.jpg *.jpeg'),
                                         ('PNG', '*.png'),
                                         ('All Files', '*.*')])
        self.fileName = filedialog.askopenfilename(**FILEOPENOPTIONS)
        
        if not self.fileName:
            return
            
        try:
            self.update_status("Loading image...")
            self.current_image_path = self.fileName
            
            # Load and display image
            image = Image.open(self.fileName)
            
            # Update image info
            img_size = os.path.getsize(self.fileName)
            img_size_kb = img_size / 1024
            width, height = image.size
            file_name = os.path.basename(self.fileName)
            
            self.image_info_label.config(text=f"File: {file_name}\n"
                                              f"Dimensions: {width} x {height}\n"
                                              f"Size: {img_size_kb:.1f} KB")
            
            # Display image
            self.display_image_in_panel(image)
            
            # Store for processing
            global mriImage
            mriImage = cv.imread(str(self.fileName), 1)
            
            # Initialize DT and set image
            if self.DT is None:
                self.DT = DisplayTumor()
            
            self.DT.readImage(image)
            
            # Create frames object if needed
            if not hasattr(self, 'FirstFrame'):
                self.FirstFrame = Frames(self, MainWindow, self.wWidth, self.wHeight, 0, 0)
                if hasattr(self.FirstFrame, 'btnView'):
                    self.FirstFrame.btnView['state'] = 'disable'
                self.listOfWinFrame = [self.FirstFrame]
            
            if len(self.listOfWinFrame) > 0:
                self.listOfWinFrame[0].readImage(image)
            
            self.viewBtn.config(state=tkinter.NORMAL)
            self.update_status(f"Loaded: {file_name}")
            
        except Exception as e:
            self.update_status(f"Error loading image: {str(e)[:50]}")
            print(f"Error loading image: {e}")
    
    def display_image_in_panel(self, image):
        """Display image in the right panel"""
        # Resize to fit panel
        display_size = (500, 500)
        image_copy = image.copy()
        image_copy.thumbnail(display_size, Image.Resampling.LANCZOS)
        
        imgTk = ImageTk.PhotoImage(image_copy)
        self.image_display_label.config(image=imgTk, text="")
        self.image_display_label.image = imgTk
    
    def show_original(self):
        """Show original loaded image"""
        if hasattr(self, 'current_image_path') and self.current_image_path:
            image = Image.open(self.current_image_path)
            self.display_image_in_panel(image)
            self.update_status("Showing original image")
    
    def show_processed(self):
        """Show processed image if available"""
        if hasattr(self, 'DT') and hasattr(self.DT, 'cv_image') and self.DT.cv_image is not None:
            processed_rgb = cv.cvtColor(self.DT.cv_image, cv.COLOR_BGR2RGB)
            pil_image = Image.fromarray(processed_rgb)
            self.display_image_in_panel(pil_image)
            self.update_status("Showing processed image")
    
    def view_image(self):
        """View image based on selected option"""
        if self.val.get() == 2:  # View Tumor Region
            self.view_tumor_region()
    
    def clear_all(self):
        """Clear all results and reset display"""
        self.result_label.config(text="No analysis performed yet", fg="black")
        self.image_display_label.config(image="", text="No Image Loaded\n\nClick 'Browse Image' to select an MRI scan")
        self.image_info_label.config(text="No image loaded")
        self.prediction_history = []
        self.update_status("Cleared all results")
        self.viewBtn.config(state=tkinter.DISABLED)
    
    def show_about(self):
        """Show about dialog"""
        about_window = tkinter.Toplevel(MainWindow)
        about_window.title("About")
        about_window.geometry("400x300")
        about_window.resizable(False, False)
        
        about_text = """
        🧠 Brain Tumor Detection System
        
        Version: 2.0
        Technology: TensorFlow, OpenCV, Tkinter
        
        This system uses deep learning to detect 
        brain tumors from MRI scans.
        
        Features:
        • Tumor detection with confidence scoring
        • Tumor region visualization
        • Medical recommendations
        
        For medical professionals only.
        Not for diagnostic use without expert review.
        """
        
        label = tkinter.Label(about_window, text=about_text, font=("Arial", 10), 
                              justify=tkinter.LEFT, padx=20, pady=20)
        label.pack()
        
        ok_button = tkinter.Button(about_window, text="OK", command=about_window.destroy,
                                   bg=self.primary_color, fg="white", padx=20)
        ok_button.pack(pady=10)
    
    def show_instructions(self):
        """Show instructions dialog"""
        instr_window = tkinter.Toplevel(MainWindow)
        instr_window.title("Instructions")
        instr_window.geometry("450x400")
        instr_window.resizable(False, False)
        
        instructions = """
        📋 How to Use:
        
        1. Click 'Browse Image' to load an MRI scan
        2. Select analysis option:
           • Detect Tumor - AI-based tumor detection
           • View Tumor Region - Visualize tumor location
           • Full Analysis - Complete diagnostic report
        3. Click 'View' to see results
        4. Use 'Clear' to reset
        
        💡 Tips:
        • Use high-quality MRI images
        • Supported formats: JPG, PNG, BMP, TIFF
        • Results are for reference only
        • Always consult medical professionals
        
        ⚠️ Disclaimer:
        This is an AI-assisted tool. Final diagnosis
        should always be made by qualified doctors.
        """
        
        label = tkinter.Label(instr_window, text=instructions, font=("Arial", 10),
                              justify=tkinter.LEFT, padx=20, pady=20)
        label.pack()
        
        ok_button = tkinter.Button(instr_window, text="Got it!", command=instr_window.destroy,
                                   bg=self.primary_color, fg="white", padx=20)
        ok_button.pack(pady=10)
    
    def check(self):
        """Perform analysis based on selected option"""
        if not hasattr(self, 'current_image_path') or not self.current_image_path:
            self.result_label.config(text="⚠️ Please browse an image first!", fg="red")
            self.update_status("Error: No image loaded")
            return
        
        if self.val.get() == 1:  # Detect Tumor
            self.detect_tumor()
        elif self.val.get() == 2:  # View Tumor Region
            self.view_tumor_region()
        elif self.val.get() == 3:  # Full Analysis
            self.full_analysis()
    
    def detect_tumor(self):
        """Perform tumor detection"""
        try:
            self.update_status("Analyzing image...")
            self.progress.start()
            
            global mriImage
            
            if mriImage is None:
                self.result_label.config(text="❌ No image loaded. Please browse an image first.", fg="red")
                self.progress.stop()
                return
            
            # Use ensemble prediction for better accuracy
            from predictTumor import predictTumor
            res = predictTumor(mriImage, use_ensemble=True)
            
            # Handle prediction result
            if isinstance(res, (list, np.ndarray)):
                prediction = float(res[0]) if len(res) > 0 else 0.0
            else:
                prediction = float(res)
            
            # Store in history
            self.prediction_history.append({
                'timestamp': datetime.now(),
                'prediction': prediction,
                'image': os.path.basename(self.current_image_path)
            })
            
            # Multi-threshold system
            HIGH_THRESHOLD = 0.75  # Lowered from 0.85 - will show tumor more often
            LOW_THRESHOLD = 0.60   # Lowered from 0.65
            NO_TUMOR_THRESHOLD = 0.40
            
            if prediction > 0.5:
                confidence = prediction * 100
            else:
                confidence = (1 - prediction) * 100
            
            # Create detailed result display
            if prediction > HIGH_THRESHOLD:
                result_text = f"🔴 TUMOR DETECTED\n\nConfidence: {confidence:.1f}%\n\n⚠️ Immediate medical consultation recommended"
                result_color = self.error_color
            elif prediction > LOW_THRESHOLD:
                result_text = f"🟠 UNCERTAIN - Possible Artifact\n\nConfidence: {confidence:.1f}%\n\n📋 Get second opinion / Repeat scan"
                result_color = self.warning_color
            elif prediction < NO_TUMOR_THRESHOLD:
                result_text = f"🟢 NO TUMOR\n\nConfidence: {confidence:.1f}%\n\n✅ Regular checkup recommended"
                result_color = self.success_color
            else:
                result_text = f"🟡 INCONCLUSIVE\n\nConfidence: {confidence:.1f}%\n\n👨‍⚕️ Consult specialist for accurate diagnosis"
                result_color = self.warning_color
            
            self.result_label.config(text=result_text, fg=result_color, font=("Arial", 10))
            
            # Show probability bar
            self.show_probability_bar(prediction)
            
            self.progress.stop()
            self.update_status(f"Analysis complete")
            
            # Print to console
            print(f"Prediction probability: {prediction:.4f}")
            print(f"Classification: {result_text.split(chr(10))[0]}")
            print("-" * 50)
            
        except Exception as e:
            self.progress.stop()
            self.result_label.config(text=f"❌ Error: {str(e)[:50]}", fg="red")
            self.update_status(f"Error: {str(e)[:50]}")
            print(f"Prediction error: {e}")
    
    def show_probability_bar(self, probability):
        """Show probability visualization"""
        prob_frame = tkinter.Frame(self.results_frame, bg=self.bg_color)
        prob_frame.pack(fill=tkinter.X, padx=10, pady=10)
        
        # Clear previous probability display
        for widget in prob_frame.winfo_children():
            widget.destroy()
        
        # Create probability bar
        prob_label = tkinter.Label(prob_frame, text=f"Probability: {probability*100:.1f}%", 
                                   font=("Arial", 9), bg=self.bg_color)
        prob_label.pack()
        
        # Custom progress bar
        canvas = tkinter.Canvas(prob_frame, width=300, height=20, bg="white")
        canvas.pack(pady=5)
        
        # Fill based on probability
        fill_width = int(300 * probability)
        if probability > 0.85:
            color = "red"
        elif probability > 0.65:
            color = "orange"
        else:
            color = "green"
        
        canvas.create_rectangle(0, 0, fill_width, 20, fill=color, outline="")
        canvas.create_rectangle(0, 0, 300, 20, outline="black")
        
        # Schedule removal after 10 seconds
        prob_frame.after(10000, prob_frame.destroy)
    
    def view_tumor_region(self):
        """View tumor region in image"""
        # Safety check
        if not hasattr(self, 'DT') or self.DT is None:
            self.result_label.config(text="❌ System not properly initialized", fg="red")
            return
        
        if not hasattr(self.DT, 'cv_image') or self.DT.cv_image is None:
            self.result_label.config(text="❌ No image loaded. Please browse an image first.", fg="red")
            self.update_status("Error: No image loaded for tumor viewing")
            return
        
        self.update_status("Displaying tumor region...")
        self.DT.displayTumor()
        self.update_status("Tumor region displayed")
    
    def full_analysis(self):
        """Perform full analysis including detection and visualization"""
        self.update_status("Performing full analysis...")
        
        # First perform detection
        self.detect_tumor()
        
        # Then show tumor region if available
        if hasattr(self, 'DT') and hasattr(self.DT, 'cv_image') and self.DT.cv_image is not None:
            # Schedule tumor display after a short delay
            MainWindow.after(2000, self.view_tumor_region)
        else:
            self.update_status("Full analysis complete (visualization not available)")

# Global variable for image
mriImage = None

# Run the application
if __name__ == "__main__":
    mainObj = Gui()