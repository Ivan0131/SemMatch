from openpyxl import load_workbook


label_vocab = ['neutral', 'entailment', 'contradiction']
wb = load_workbook("pred_report.xlsx")
ws = wb['examples']
with open("submission.csv", 'w', encoding='utf-8') as csv_file:
    csv_file.write("pairID,gold_label\n")
    for row in ws.iter_rows():
        if row[0].value == "index":
            continue
        csv_file.write("%s,%s\n"%(row[0].value, label_vocab[int(row[3].value)]))

