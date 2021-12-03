# ===================================================================================================================================
# Code: Calculation of average heart rate
# ===================================================================================================================================

# -----------------------------------------------------------------------------------------------------------------------------------
# Importing Libraries 

import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import cv2

facecascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# -----------------------------------------------------------------------------------------------------------------------------------
# Functions

def get_signal(cap):
    """

    Parameters
    ----------
    cap : VideoCapture
        Retrieves the information transmitted by the camera.

    Returns
    -------
    signal : Array of float64
        Matrix of the evolution of pixels over time.

    """
    # counter
    i = 0
    # if state = False the while loope stops 
    state = True
    # matrix of pixel evolution over time
    signal = np.zeros((n , nb_pixels_l * nb_pixels_h + nb_pixel_bruit ** 2 + 1))
    wait = "Wait"
    while state :

        # image acquisition
        check , frame = cap.read()
        # image symmetry
        frame = cv2.flip(frame , 1)
        # gray shade transformation
        gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)

        # face detection
        faces = facecascade.detectMultiScale(gray , 1.3 , 5)

        if len(faces) != 0 :
            for (x , y , w , h) in faces :
                if h > 0 and w > 0 :
                    # face contour
                    cv2.rectangle(frame , (x , y) ,
                                  (x + w , y + h) , (255 , 0 , 0) , 2)

                    # pixel selection outline
                    cv2.rectangle(frame , (x + w // 2 - nb_pixels_l // 2 , y + h // 10 - nb_pixels_h // 2) ,
                                  (x + w // 2 + nb_pixels_l // 2, y + h // 10 + nb_pixels_h // 2) , (255 , 255 , 0) , 2)
                    pixels = gray[y + h // 10 - nb_pixels_h // 2: y + h // 10 + nb_pixels_h // 2 ,
                             x + w // 2 - nb_pixels_l // 2: x + w // 2 + nb_pixels_l // 2]
                    # resizing to be able to create the signal matrix
                    # at time i, we add the pixels in the form of line vectors
                    signal[i, :nb_pixels_h * nb_pixels_l] = pixels.ravel()

                    # pixel in the center of the forehead
                    cv2.rectangle(frame , (x + w // 2 , y + h // 10) ,
                                  (x + w // 2 + 1 , y + h // 10 + 1) , (255 , 0 , 255) , 2)

                    # image noise pixel
                    cv2.rectangle(frame , (20 , 20) ,
                                  (20 + nb_pixel_bruit , 20 + nb_pixel_bruit) , (255 , 255 , 255) , 2)
                    bruit = gray[20 + nb_pixel_bruit , 20 + nb_pixel_bruit]
                    signal[i , nb_pixels_h * nb_pixels_l:-1] = bruit.ravel()

                    # the last noise value is the average of the image
                    signal[i , -1] = gray.mean()

                    i += 1

        # picture display
        font = cv2.FONT_HERSHEY_SIMPLEX # font for the date
        text = 'Date : ' + str(datetime.datetime.now())
        frame = cv2.putText(frame , text , (10 , 60) , font , .7 , (0 , 255 , 255) , 2 , cv2.LINE_AA)# Adding the date
        frame = cv2.putText(frame , "Please don't move" , (10 , 80) , font , .7 , (0 , 255 , 255) , 2 , cv2.LINE_AA)# Adding the date
        
        key = cv2.waitKey(1)
        if key == ord('q') : # condition which allows the program to be stopped by pressing the "q" key
            break

        # progress of pixel acquisition
        if i / n * 100 % 5 == 0 :
            wait = f'Please wait : {int(i / n * 100)} %'
        
        frame = cv2.putText(frame , wait , (10 , 100) , font , .7 , (0 , 255 , 255) , 2 , cv2.LINE_AA)# Adding the date
        cv2.imshow('Heart Health' , frame)
        # once the matrix is completed we stop the video loop
        if i == n :
            state = False
            cv2.destroyAllWindows()

    return signal


def fourier(signal , n , step) :
    """

    Parameters
    ----------
    signal : Array of float64
        Matrix of the evolution of pixels over time.
    n : int
        number of measurements.
    step : float
        sampling step.

    Returns
    -------
    freq : Array of float64
        frequency of sampling points per minute.
    sp :  Array of float64
        discrete Fourier transformation of the signal.

    """
    # step in second thus freq is in Hz
    freq = np.fft.fftfreq(n , d = step)
    sp = np.abs(np.fft.fft(signal , axis=0))

    # converting frequency to bpm
    freq *= 60
    return freq , sp


def filtre(freq , sp , f1 , f2) :
    """
    
    Parameters
    ----------
    freq : Array of float64
        frequency of sampling points in bpm.
    sp : Array of float64
        discrete Fourier transformation of the signal.
    f1 : int
        minimum frequency allowed.
    f2 : int
        maximum frequency allowed.

    Returns
    -------
    freq : Array of float64
        frequency of sampling points in Hz after being filtered.
    sp : Array of float64
        Discrete Fourier transformation of the signal after being filtered.

    """
    mask = [i and j for i , j in zip(freq > f1 , freq < f2)]
    freq = freq[mask]
    sp = sp[mask]

    return freq , sp


def traitement(signal , sp) :
    """

    Parameters
    ----------
    signal : Array of float64
        Matrix of the evolution of pixels over time.
    sp_filtre : Array of float64
        Discrete Fourier transformation of the signal after being filtered.

    Returns
    -------
    temporel : Array of float64
        Vector a series of three signal vectors of noises 1 and 2 in the time domain.
    frequentiel : Array of float64
        Result of the temporal Fourier transform.

    """
    # we average all the measurements
    pixels = signal[: , : nb_pixels_l * nb_pixels_h].mean(axis = 1)
    sp_pixels = sp[: , : nb_pixels_l * nb_pixels_h].mean(axis = 1)

    bruit_1 = signal[: , nb_pixels_l * nb_pixels_h : -1].mean(axis = 1)
    sp_bruit_1 = sp[: , nb_pixels_l * nb_pixels_h : -1].mean(axis = 1)

    bruit_2 = signal[: , -1]
    sp_bruit_2 = sp[: , -1]

    # we normalize all the results between 0 and 1
    pixels = pixels / pixels.max()
    sp_pixels = sp_pixels / sp_pixels.max()

    bruit_1 = bruit_1 / bruit_1.max()
    sp_bruit_1 = sp_bruit_1 / sp_bruit_1.max()

    bruit_2 = bruit_2 / bruit_2.max()
    sp_bruit_2 = sp_bruit_2 / sp_bruit_2.max()

    temporel = np.array([pixels, bruit_1, bruit_2]).T
    frequentiel = np.array([sp_pixels, sp_bruit_1, sp_bruit_2]).T
    
    bpm = freq[max(enumerate(sp_pixels) , key = lambda x : x[1])[0]]
 
    return temporel , frequentiel , bpm


def show(freq , temporel , frequentiel) :
    """

    Parameters
    ----------
    freq : Array of float64
        frequency of sampling points in bpm.
    temporel : Array of float64
       Vector a series of three signal vectors of noises 1 and 2 in the time domain.
    frequentiel : Array of float64
        Result of the temporal fourier transform.

    Returns
    -------
    Graphic display.

    """
    color = ['b' , 'g' , 'm']
    legend_1 = ['pixels' , 'noise_1' , 'noise_2']
    legend_2 = ['sp_pixels' , 'sp_noise_1' , 'sp_noise_2']
    plt.figure("Graphic representations")
    
    # -------------------------- time domain --------------------------
    
    plt.subplot(211)
    plt.plot(np.linspace(0 , end - start , n) , temporel)
    plt.xlabel('time (in s)')
    plt.legend(legend_1)

    # ------------------------- frequency domain -----------------------
    
    plt.subplot(212)
    plt.plot(freq , frequentiel)
    plt.xlabel('fréquency (in bpm)')
    plt.legend(legend_2)

    plt.show()


# -----------------------------------------------------------------------------------------------------------------------------------
# Constants

L = 640             # camera return length
H = 480             # camera return height
n = 1000            # number of measurements (duration ~ = 30s for n = 1000)
nb_pixels_l = 100       # measuring surface (even number)
nb_pixels_h = 6
nb_pixel_bruit = 10     # square length (10 * 10 pixels)
f1 = 50  # we keep that the frequencies between f1 and f2
f2 = 180

# video stream initialization
cap = cv2.VideoCapture(0)
cap.set(3 , L)
cap.set(4 , H)

# -----------------------------------------------------------------------------------------------------------------------------------
# Main program

# we start the stopwatch for the sampling frequency
start = time.time()

# we gather the pixels of the face over time
signal = get_signal(cap)

# we stop the chrono and calculate the sampling step
end = time.time()
step = (end - start) / n

# Fourier transform of signals and pixel noise
freq , sp = fourier(signal , n , step)

# filter between f1 and f2 in bpm
freq, sp = filtre(freq , sp , f1 , f2)

# signal processing
temporel , frequentiel, bpm = traitement(signal , sp)

# result display
show(freq , temporel , frequentiel)


# heart rate display on the console
print(f'\n heart rate : {int(bpm)} bpm')