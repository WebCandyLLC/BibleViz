import json

file_name = "C:\\Users\\webca\\AppData\\Roaming\\nltk_data\\corpora\\gutenberg\\bible-kjv-conversion.txt"  # Update with the actual file name
output_file = "outputV2.json"  # Update with the desired output file name

with open(file_name, "r") as file:
    data = file.read()

books_data = data.strip().split('\n\n')  # Split data into separate books

result = {}

for book_data in books_data:
    word_count = 1
    book_lines = book_data.strip().split('\n')
    book_name = book_lines[0].strip()
    verses = book_lines[1:]

    book_dict = {}

    current_chapter = None
    prev_chapter = "1"
    for verse in verses:
        parts = verse.split(' ')

        # Check if the line contains a chapter:verse entry
        if len(parts[0].split(':')) == 2:
            chapter, verse_number = parts[0].split(':')
            verse_content = ' '.join(parts[1:])

            # Update current chapter
            current_chapter = chapter

            if current_chapter not in book_dict:
                book_dict[current_chapter] = { 'word_count_str': word_count, 'word_count_end': 0, 'verses': [], 'verses_str_pos': []}

            if current_chapter != prev_chapter:
                book_dict[prev_chapter]["word_count_end"] = word_count - 1
            
            book_dict[current_chapter]["verses"].append(verse_content)
            book_dict[current_chapter]["verses_str_pos"].append(word_count)
            word_count += len(verse_content.split(' '))

        '''else:
            # UNUSED
            # Line does not contain a chapter:verse entry, treat it as part of the previous verse
            verse_content = ' '.join(parts)

            if current_chapter is not None:
                book_dict[current_chapter][-1] += ' ' + verse_content'''

        prev_chapter = current_chapter

    book_dict[current_chapter]["word_count_end"] = word_count - 1    
    result[book_name] = book_dict

# Write the result to the output file
with open(output_file, "w") as output:
    json.dump(result, output)

print("Data written to", output_file)

