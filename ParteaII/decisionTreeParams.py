import subprocess

outputPath = "accuracy.txt"
with open(outputPath, "w") as file:
    file.write("entropy\n")
    for i in range(2, 20):
        file.write(f"\n{i}\n")
        result = 0
        for j in range(0, 5):
            command = ["python3", "parteaII.py", "entropy", str(i)]
            result += float(subprocess.run(command, capture_output=True, text=True).stdout)
        file.write(f"{result / 5.0}")

    file.write("\ngini\n")
    for i in range(2, 20):
        file.write(f"\n{i}\n")
        result = 0
        for j in range(0, 5):
            command = ["python3", "parteaII.py", "gini", str(i)]
            result += float(subprocess.run(command, capture_output=True, text=True).stdout)
        file.write(f"{result / 5.0}")

    file.write("\nlog_loss\n")
    for i in range(2, 20):
        file.write(f"\n{i}\n")
        result = 0
        for j in range(0, 5):
            command = ["python3", "parteaII.py", "log_loss", str(i)]
            result += float(subprocess.run(command, capture_output=True, text=True).stdout)
        file.write(f"{result / 5.0}")
