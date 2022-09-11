# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 09:06:08 2022

@author: Riccardo Sala
"""


# list to store file lines
lines = []
# read file
with open(r"C:\Users\leonardo\Desktop\Python\Python_projects\NeuralChatbot\MioNeuralCB\data\movie_lines.txt", "r") as fp1:
    # read an store all lines into list
    lines = fp1.readlines()

# Write file
with open(r"C:\Users\leonardo\Desktop\Python\Python_projects\NeuralChatbot\MioNeuralCB\data\movie_lines.txt", 'w') as fp1:
    # iterate each line
    for number, line in enumerate(lines):
        # delete line 5 and 8. or pass any Nth line you want to remove
        # note list index starts from 0
        if number not in range (150000,304710):
            fp1.write(line)

fp1.close()

print('fatto lines')
# list to store file lines
lines = []
# read file
with open(r"C:\Users\leonardo\Desktop\Python\Python_projects\NeuralChatbot\MioNeuralCB\data\movie_conversations.txt", "r") as fp2:
    # read an store all lines into list
    lines = fp2.readlines()

# Write file
with open(r"C:\Users\leonardo\Desktop\Python\Python_projects\NeuralChatbot\MioNeuralCB\data\movie_conversations.txt", 'w') as fp2:
    # iterate each line
    for number, line in enumerate(lines):
        # delete line 5 and 8. or pass any Nth line you want to remove
        # note list index starts from 0
        if number not in range (40000, 83097):
            fp2.write(line)
            
            
fp2.close()
print('fatto convers')



#%%


file1 = open(r"C:\Users\leonardo\Desktop\Python\Python_projects\NeuralChatbot\MioNeuralCB\data\movie_lines.txt", "r")
line_count1 = 0.
for line in file1:
    if line != "\n":
       line_count1 += 1.
print(line_count1)

file1.close()



file2 = open(r"C:\Users\leonardo\Desktop\Python\Python_projects\NeuralChatbot\MioNeuralCB\data\movie_conversations.txt", "r")
line_count2 = 0.
for line in file2:
    if line != "\n":
       line_count2 += 1.
print(line_count2)


file2.close()




