from openpyxl import load_workbook


# Opens a single sheet for reading and returns dict object
def read_worksheet(filename, sheet_name):
    wb = load_workbook(filename, read_only=True)
    ws = wb[sheet_name]
    column_labels = next(ws.rows)
    dataset = []
    classes = set()
    for row in ws.rows:
        new_entry = {}
        classes.add(row[6].value)
        for i in range(11):
            new_entry[column_labels[i].value.lower().replace(" ", "_")] = str(row[i].value)
        dataset.append(new_entry)
    wb.close()
    return dataset[1:], classes


# TODO: Iterate through every row (message) in dataset and pre-process it
def preprocess_data(dataset):
    for entry in dataset:
        entry["message"] = entry["message"].strip()


if __name__ == "__main__":
    crew_data, classes = read_worksheet("dataset.xlsx", "CREW data")
    # print(classes1)