def verify():
    cakes = [[int(val) for val in line.rstrip().split(" ")]
             for line in open("asgmt3/bakery.txt", 'r')]
    starts = [[float(val) for val in line.rstrip().split(" ")]
              for line in open("asgmt3/starts.txt", 'r')]
    starts = sorted(starts, key=lambda x: x[1])

    for start in starts:
        print(
            f"Cake{int(start[0])} starts at {start[1]} and ends at {cakes[int(start[0])][3] + start[1]}")
