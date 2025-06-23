# Music-Generation
Music Generation using XLSTM

# Introduction :

Implemented an xLSTM-based model for sequential music generation. 

Generated coherent and musically plausible compositions across various musical styles. 

Evaluated the xLSTM model's performance against established baselines in terms of musical quality and novelty. 

Explored xLSTM's capability in capturing long-term musical dependencies and structural elements. 

Developed a user-friendly interface for generating and interacting with the music.



# Steps:

1. Create a folder by the name "data"
2. Inside it, create a folder named "blues" and "pop"
3. Download midi files from google of the respective genre in (atleast 100 each and avoid vocal songs)
4. Save the songs in the respective folder
5. Check the number of epochs to be 50
6. run the programme with command eg. `python3 chk1.py blues --retrain`
7. wait for 3-4 hours
8. oce weights file generated `python3 chk1.py blues`
8. once weights file generated `python3 chk1.py blues`
