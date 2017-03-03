# be-proj-searchengine

## VESIT Student's Project Guidance

## Using Git
1. Download [Git](https://git-scm.com/download/win) for windows
2. Follow this [Tutorial](https://help.github.com/desktop/guides/getting-started/)

## Current Code Strcture:

1. home.py -> Flask fontend
2. extract_books.py -> This is used to extract book title and author's name
3. language_model.py -> Taking the search query it return corrected grammar words
4. sample_2.py -> Does clustering
5. spell_check.py -> Does spell checking
6. text_segment.py -> Take search query and return a segmented query

## Code Review

1. 27th Jan 2017:
    - home.py should be renamed to server.py
    - Add basic documentation on top of each file tell what exactly is the purpose of the file
    - Create a function `def extract_title_and_author(file_name)` in extract_books.py
    - Coding style is not consistent function names. E.g. `Pword` should be converted to `pword`
    - In language_model.py document the flow of things. The sequence in which functions are called
    - For sample_2.py
        - Rename sample_2.py should be renamed to cluster_text.py
        - Specify algorithm in text comments i.e. Step 1, Step 2, Step 3 etc..
        - Implement **one algorithm from scratch** even simple KNN/K-Mean algorithm
        - Create a variable called THRESHOLD on top of the file which will allow you to control output and run in different modes
    - Create an excel sheet of book names and category/genre. Basis this we will decide THRESHOLD values
    - Once this is done we will create another dataset of 100 books and run the exact same algo and setting to see how it performs
    - **Doubt Resolution:**
        1. What should be ideal thresold value for flat clustering? (TBD on above steps)
        2. How will we use book contents for clustering?  (We will use contents of the books for clustering as opposed to just titles and names)
    - spell_check.py and text_segment should have similar comments

Note : Foolow thw steps accordingly. 
