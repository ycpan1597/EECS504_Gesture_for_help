Welcome to the EECS504 Final Project repository! Each python script and their intended purpose is listed below
<ul>
    <li>
        runnable files
        <ul>
            <li>data_preparation.py: used to create dataset with manual bounding box and threshold</li>
            <li>parameter_test.py: used to test a set of parameters or perform X-fold validation;<br>saves a json file of useful metrics</li>
            <li>video_annotation.py: used to annotate a video frame-by-frame with an ML-model</li>
        </ul>
    </li>
    <li> function files
        <ul>
            <li>cv2_tutorial.py: tutorials of on how to read/write/annotate images/videos using OpenCV</li>
            <li>HandGestureDataset.py: custom dataset class that inherits torch.utils.data.Dataset</li>
            <li>ml_script.py: contains the "train" function called by "parameter_test.py" </li>
            <li>utils.py: other useful functions include dataset preprocessing (eg. resizng)</li>
            <li>models.py: includes our simple 2-layer CNN architecture</li>
        </ul>
    </li>
</ul>