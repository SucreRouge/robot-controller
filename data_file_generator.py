from numpy import *
import csv

fields = []
data = []

try:
    with open("Vx_Vy_sonar_reading_March10e_1500.csv", 'r') as rawData:
        reader = csv.reader(rawData)
        fields = reader.next()

        for row in reader:
            data.append(row)

        n = reader.line_num - 1
except IOError as e:
    print "raw data file absent in current directory"

m = len(fields)

cleanData = ones((n, m + 3))
cleanData[0][m] = 0.0
cleanData[0][m + 1] = 0.0
for i in range(m):
    cleanData[0][i] = data[0][i]

for i in range(1, n):
    cleanData[i][m] = cleanData[i-1][0]
    cleanData[i][m + 1] = cleanData[i-1][1]
    for j in range(m):
        cleanData[i][j] = data[i][j]

minCD = zeros(m+3)
maxCD = zeros(m+3)

for i in range(m+3):
    minCD[i] = min(cleanData[:,i])
    maxCD[i] = max(cleanData[:,i])

mappedData = ones((shape(cleanData)))
for i in range(n):
    for j in range(m+2):
        mappedData[i][j] = (cleanData[i][j] - minCD[j])/(maxCD[j] - minCD[j])

savetxt("data21.csv", cleanData, delimiter = ',')
savetxt("data22.csv", mappedData, delimiter = ',')
