## Using Clinical Notes with Time Series Data for ICU Management

This is code for the paper, "Using Clinical Notes with Time Series Data for ICU Management" 
at EMNLP 2019 by Swaraj Khadanga, Karan Aggarwal, Shafiq R. Joty, Jaideep Srivastava.

This code has been modified by Matt Whitehill, Jacob Peplinski, and Yue Guo.

The purpose of this project is to explore the use of textual information in three ICU prediction tasks.
All experiments are based on the MIMIC-III clinical database which is required before this code can be used.

# To Run This Code
- Apply for access to the MIMIC-III Database [here](https://mimic.physionet.org/).
  Note that getting approved can take upwards of two weeks.
- Clone this repository to your machine.
- Clone [this benchmark repository](https://github.com/YerevaNN/mimic3-benchmarks).
  It contains benchmark code and models for MIMIC-III. Make sure the directory is cloned 
  **inside of this project folder**.
- Preprocess the clinical data by following the README in the benchmark repository. 
  This step will take several hours to complete.
- In this repository, run extract_notes.py to preprocess the clinical notes.
- In this repository, run extract_T0.py to align the clinical notes to time-series data.  


## Configuration
1. Update all paths and configuration in config.py file.

## Models
- For IHM run ihm_model.py file under tf_trad.

    Number of train_raw_names:  14681 <br>
    Succeed Merging:  11579 - Model will train on this many episodes as it contains text. <br>
    Missing Merging:  3102 - These texts don't have any text for first 48 hours. 

- For Decompensation, run decom_los_model.py file under tf_trad.

    Text Not found for patients:  6897 <br>
    Successful for patients:  22353

- Length of Stay, run decom_los_model.py file under tf_trad.

    Successful for episodes for training:  22353