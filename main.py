import csv

with open('EURUSD_M1.csv') as f:
    reader = list(csv.reader(f))
    for row in reader[:10]:
        row.pop(0)
        print(row)
