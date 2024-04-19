
import json


def make_html_from_json():
    def html_from_json(path, json_data):
        blocks = json_data["lines"]
        data = []
        for block in blocks:
            block['bbox'].pop(0)
            # Calculate grid row and column based on bbox values
            # This is a simple interpretation and might need adjustments for more complex layouts
            row = int(block["bbox"][1] * 70) + 1
            col = int(block["bbox"][0] * 40) + 1
            text = block["text"].rstrip('\n')
            data.append((row, col, text))

        # Sort data by row and then by column
        data.sort(key=lambda x: (x[0], x[1]))

        # Creating the HTML table with colspan
        table_html = '<table border="1"> '
        current_row = 0
        last_col = 0
        for row, col, text in data:
            if row != current_row:
                if current_row != 0:
                    table_html += '</tr> '
                table_html += '<tr>'
                current_row = row
                last_col = 0
            colspan = col - last_col - 1
            if colspan > 0:
                table_html += f'<td colspan="{colspan}"></td>'
            table_html += f'<td>{text}</td>'
            last_col = col
        table_html += '</tr> </table>'

        with open(path, 'w') as file:
            file.write(table_html)

    with open('dataset.jsonl', 'r') as file:
        for line in file:
            # Parse each line as a JSON object
            json_obj = json.loads(line)
            fpath = json_obj['file_path']
            pages = json_obj['ocr']['pages']
            for page in pages:
                id = page['page_id']
                id = '' if len(pages) < 2 else '-'+str(id)
                jpath = fpath.replace(
                    '.pdf', id+'.html').replace('pdfs', 'vis')
                html_from_json(jpath, page)


make_html_from_json()
