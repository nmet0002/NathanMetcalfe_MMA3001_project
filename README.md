# Nathan Metcalfe's MMA3001 project
This project is designed to help machinists model and estimate flank wear over time on CNC milling machines
using a neural network model under various conditions. It can also be used to estimate the time at which the
tool will reach an inputted threshold flank wear to be replaced

# Instructions
1. Download the data

   The publicly available NASA dataset to train the model can be downloaded from https://data.nasa.gov/dataset/milling-wear
   The zip file will need to be extracted so that mill.mat can be accessed

2. Clone repository (Only if running notebook in Colab on the web)

   In order to access the functions file, the repository must be cloned using the provided first cell, or uploaded
   manually

3. Specify Data Filepath

   At the top of the cell labelled "Model setup/training", you must specify the filepath to mill.mat
   If storing the data file in Google Drive and accessing through Colab, run the cell to connect to your
   Google Drive

4. Run the notebook!

   After training the model and viewing the graphs of the result, there are two different cells with functions that
   can be used to evaluate the model in different ways

5. EXTENSION: Vibration Data

   If you wish, there is another model that can be trained with vibration data included. This isn't something
   usually accessible in advance of machining, but could be used to estimate wear at the current moment in time!

Check out the development folder to view the rough working out files needed to figure out importing and cleaning MAT data,
inspecting the data, and creating/testing the models under various different settings!

# AI Statement
Google Gemini 2.5 LLM that is built into Google Colab and ChatGPT have been used to help setup the Git repository
and generate portions of code. All prompts were provided by myself, and I have made sure to fully
check all outputs with my professional engineering judgement, as outlined by Engineers Australia. 
Further AI reflection can be found in the project report.
