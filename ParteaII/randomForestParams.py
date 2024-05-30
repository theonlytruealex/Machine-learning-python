import subprocess

outputPath = "accuracy.txt"
with open(outputPath, "w") as file:
    file.write("entropy\n")
    for i in range(50, 200, 10):
        file.write(f"\n{i}\n")
        result = 0
        for k in range(5, 20):
            for j in range(0, 5):
                command = ["python3", "parteaII.py", str(i), "entropy", str(k)]
                result += float(subprocess.run(command, capture_output=True, text=True).stdout)
            file.write(f"{k}: {result / 5.0}")

    file.write("\ngini\n")
    for i in range(50, 200, 10):
        file.write(f"\n{i}\n")
        result = 0
        for k in range(5, 20):
            for j in range(0, 5):
                command = ["python3", "parteaII.py", str(i), "gini", str(k)]
                result += float(subprocess.run(command, capture_output=True, text=True).stdout)
            file.write(f"{k}: {result / 5.0}")
