# RSE SEMG Dataset (Finger-Focus)
* Participants: 10 subjects.
* Gestures (4 Classes): Index Finger Extension, Middle Finger Extension, Cylindrical Grip, Closed Grip, and Rest
* Data Density: Focusing on precision-based finger movements.
* Data Format: Retains raw Analog-to-Digital Converter (ADC) counts.
* Validation Strategy: All recordings from subject 10 were reserved for testing, ensuring zero information leakage during the evaluation.

The Myo Armband by Thalmic labs was used for data acquisition, the armband consists of 8 Surface EMG sensor units and 8 sensors read data every 5ms:
- This data is stored in a CSV file with the timestamp. The sensor names are labeled as well. Each user wore the armband on their forearm and performed the 5 different gestures.100 instances were collected for each user for each of the gestures. one instance included performing the gesture within a window of 2 seconds followed by a rest window of 2 seconds. 
- This pattern of flexion and extension gestures is performed to acquire the data. Each session was restricted to a batch of 25 instances considering the fatigue that sets in after constant flexion and extension gestures.

The gesture data corresponding to each user are organized into 10 folders labeled as "User #", Inside each User folder, the file corresponding to each gesture are organized in a folder labeled with the gesture names "Index Finger Extension", "Middle Finger Extension", "Cylindrical Grip", "Closed Grip", and "Rest". Each CSV file is labeled as "u#-gesture-name-set-#.csv". 