choose_problem_path_prompt_template = """
    * You are given a student question about an assignment problem.
    * You also have access to all assignment questions in a directory structure.
    * You need to select the directory path containing the assignment question mentioned by the student.
      * First, find the closest assignment directory that best matches the assignment mentioned in the student's question.
      * Then, find the question directory within the chosen assignment directory that best matches the assigment question mentioned in the student's question. You can ignore question subparts or subsections; for example, question 1.1 will be contained in the path for question 1.
      * If there is no approximate match for the assignment question, select the path to the assignment directory.
    Format the output as follows: {"reasoning": "your thought process of how you processed student's question and selected the path", "selected_path": "path to the directory"}, do not include any other formatting.
"""

def get_choose_problem_path_prompt(paths: str, question_info: str) -> list:
    return [{'role': 'user', 'content': f"Student question: {question_info}, Paths: {paths}\n\n{choose_problem_path_prompt_template}"}]


summarize_conversation_prompt = [
    {'role': 'system', 'content': """
     You are a helpful assistant that summarizes inputs. 
     You will receive a list of dictionaries as input, representing a conversation between a Student and a TA (Teaching Assistant).
     The last entry will always have 'role': 'Student'. 
     Your job is to retrieve the text of the last query from the Student, and to summarize the relevant conversation history needed to understand the query.

     Here is an example:

     Input:
     [{'role': 'Student', 'text': 'I'm a bit confused about nested functions. I went over the section in the textbook but it just doesn't make a lot of sense to me.', 'image context': ''},
     {'role': 'TA', 'text': 'Can you please provide a bit more context about the part that confuses you?', 'image context': ''},
     {'role': 'Student', 'text': 'Nested Function Confusion\n\nYeah, I'm struggling to understand how the return values work in nested functions and also nested called are really confusing for me. For example, the make_adder(3)(5) call in the textbook confuses me...', 'image context': ''}]

     Output:
     I need help understanding how return values and nested calls work in nested functions. Specifically, I would like an explanation about make_adder(3)(5).
     """}]


def get_summarize_conversation_prompt(conversation: list) -> list:
    return summarize_conversation_prompt + [{'role': 'user', 'content': str(conversation)}]


assignment_1_system_prompt = """You will simulate the role of a teaching assistant for an undergraduate computer science course, answering assignment-related questions. 
(1) Your responses should be clear, helpful, and maintain a positive tone.
(2) Your goal is to guide the student to the correct answer. Do not give away the solution to the question or directly reference the solutions; only provide hints and explanations.
(3) Aim for conciseness and clarity.
(4) Use the provided assignment instructions and solutions and similar historical question-answer pairs as your primary source of information.
(5) If you need additional context and information, ask the student for clarification.
(6) If the question is difficult to answer based on the provided context, reply, "Sorry, I do not know. Please wait for a staff member's response." """
assignment_1_few_shot_1_user = """Here are the relevant sections from the assignment solutions to guide your response:
==========================================
Assignment solutions
==========================================
#### Lab 1 Question 1
Write a recursive function that takes in a positive integer n and determines if its digits contain two adjacent 8s (that is, two 8s right next to each other). Do not use for or while.

Hint: Start by coming up with a recursive plan: the digits of a number have double eights if either (think of something that is straightforward to check) or double eights appear in the rest of the digits.
    \"\"\" Returns whether or not n has two digits in row that
    are the number 8. Assume n has at least two digits in it.

    >>> double_eights(1288)
    True
    >>> double_eights(880)
    True
    >>> double_eights(538835)
    True
    >>> double_eights(284682)
    False
    >>> double_eights(588138)
    True
    >>> double_eights(78)
    False
    >>> from construct_check import check
    >>> # ban iteration
    >>> check(LAB_SOURCE_FILE, 'double_eights', ['While', 'For'])
    True
    \"\"\"
# BEGIN SOLUTION
last, second_last = n % 10, n // 10 % 10
if last == 8 and second_last == 8:
    return True
elif n < 100:
    return False
return double_eights(n // 10)
# END SOLUTION
==========================================

Here are similar historical question-answer pairs to guide your response:
==========================================
Retrieved historical QA
==========================================
Conversation History and Student question:
[
    {"role": "Student", "text": "Lab 1 Question 1: My function correctly identifies when the last two digits are 88, but it seems to return incorrect results for other cases. It also doesn't stop when the number is too small to have two digits. Here's my code:\nif n % 100 == 88:\nreturn True\nelse:\nreturn double_eights(n % 10)", "image context": ""}
]
TA's response: "You're close! Your approach to checking if the last two digits are 88 is good, but you're missing a base case that handles when the number is too small to have two digits. 
You should add a base case to return False when n is less than 10 because, at that point, it's impossible to have two adjacent 8s. Also, make sure the function properly continues checking for 88 throughout the number. 
Try updating your function with these changes, and it should work as expected. Keep it up!"
==========================================

==========================================
Conversation History and Student question:
[
    {"role": "Student", "text": "I'm having trouble understanding how to approach the recursive logic for checking adjacent 8s. Should I be checking all digits of the number at once or just focusing on the last two digits?", "image context": ""},
    {"role": "TA", "text": "Good question! When dealing with recursion, it's helpful to focus on breaking the problem down into smaller parts. For this problem, instead of checking all the digits at once, try focusing on just the last two digits at each recursive step. If the last two digits are not both 8, you can strip off the last digit and continue checking the rest of the number. This way, your function only needs to focus on pairs of digits as it works through the number.", "image context": ""},
    {"role": "Student", "text": "Debugging Lab 1 q1: I'm not sure why my solution is wrong and failing the tests. here's my code:\nlast\nif last == 8 \nreturn True\nelif n < 100:\nreturn False\nreturn double_eights(n // 10)", "image context": ""}
]
==========================================
Given the conversation between the student and the TA, answer the most recent student question concisely based on the provided assignment solutions and historical question-answer pairs. Do not repeat what has already been said. Do not give away or directly refer to the solutions."""
assignment_1_few_shot_1_assistant = """Hi! That is a great question!
Your code and logic make sense. However, let's double check the instructions. 
The prompt asks us to look at pairs of "8"s. We are looking for whether or not 
n has two digits in row that are the number 8. Right now, the code is only looking 
at a single digit at a time. How can we adjust this so we are looking at two digits 
at a time. I hope this hint was helpful. Feel free to follow up if you have further questions!"""


def get_first_assignment_prompt(processed_conversation: str, retrieved_qa_pairs: str,
                                retrieved_docs_manual: str) -> list:
    curr_prompt = f"""Here are the relevant excerpts from the assignment solutions to guide your response:
    ==========================================
    {retrieved_docs_manual}
    ==========================================

    Here are similar historical question-answer pairs to guide your response:
    ==========================================
    {retrieved_qa_pairs}
    ==========================================

    ==========================================
    Conversation History and Student question:
    {processed_conversation}
    ==========================================
    Given the conversation between the student and the TA, answer the most recent student question concisely based on the provided assignment solutions and historical question-answer pairs. Do not repeat what has already been said. Do not give away or directly refer to the solutions."""
    return [
        {"role": "system", "content": assignment_1_system_prompt},
        {"role": "user", "content": assignment_1_few_shot_1_user},
        {"role": "assistant", "content": assignment_1_few_shot_1_assistant},
        {"role": "user", "content": curr_prompt}
    ]


assingment_2_system_prompt = """Given a student's question and a potential answer, please modify the answer according to the following guidelines:
1. Revise the answer to make it more concise.
2. Remove any solutions provided in the original answer, leaving only hints and guiding explanations.
3. Encourage the student to ask follow-up questions if they need further clarification."""
assignment_2_few_shot_1_user = """Conversation History and Student question:
==========================================
[
    {"role": "Student", "text": "questionon project 1 question 1: I'm calling the dice function by returning dice but nothing is happening", "image context": ""}
]
==========================================

Answer:
==========================================
It’s crucial to remember that in order to run a function that is passed in as an argument, we need to call it!
We can do this by calling dice(). Because dice() does not take in arguments, we do not pass any arguments into the function.
=========================================="""
assignment_2_few_shot_1_assistant = """You need to call the actual function instead of making a reference to it.
Please try making this adjustment and let me know if you have any further questions!"""


def get_second_assignment_prompt(processed_conversation: str, first_answer: str) -> list:
    curr_prompt = f"""Conversation History and Student question:
    ==========================================
    {processed_conversation}
    ==========================================

    Answer:
    ==========================================
    {first_answer}
    =========================================="""
    return [
        {"role": "system", "content": assingment_2_system_prompt},
        # {"role": "user", "content": assignment_2_few_shot_1_user},
        # {"role": "assistant", "content": assignment_2_few_shot_1_assistant},
        {"role": "user", "content": curr_prompt}
    ]


content_system_prompt = """
You will simulate the role of a teaching assistant for an undergraduate data science course, answering student questions based on the provided excerpts from the course notes and historical question-answer pairs.
(1) Your responses should be clear, helpful, and maintain a positive tone.
(2) Aim for conciseness and clarity.
(3) Use the excerpts from the course notes and historical question-answer pairs as your primary source of information.
(4) If you need additional context and information, ask the student for clarification.
(5) If the question is difficult to answer based on the provided context, reply, "Sorry, I do not know. Please wait for a staff member's response." """
content_few_shot_1_user = """Here are the sections from the course notes to guide your response:
==========================================
Retrieved document
==========================================
::: {.example step="3"}
chinese = \[\'coin\', \'string\', \'myriad\'\] suits = chinese
suits.pop() suits.remove(\'string\') suits.append(\'cup\')
suits.extend(\[\'sword\', \'club\'\]) suits\[2\] = \'spade\'
suits\[0:2\] = \[\'heart\', \'diamond\'\]
:::

Retrieved document
==========================================
\>\>\> suits.pop() \# Remove and return the final element \'myriad\'
\>\>\> suits.remove(\'string\') \# Remove the first element that equals
the argument

Three more suits were added (they evolved in name and design over time),

\>\>\> suits.append(\'cup\') \# Add an element to the end \>\>\>
suits.extend(\[\'sword\', \'club\'\]) \# Add all elements of a sequence
to the end

==========================================

Here are similar historical question-answer pairs to guide your response:
==========================================
Retrieved historical QA
==========================================
Conversation History and Student question:
[
    {"role": "Student", "text": "Append vs Extend: What's the difference between append vs extend?", "image context": ""}
]
TA's response: The following code block demonstrates the difference between append and extend: 
x = [1, 2, 3] 
x.append([4, 5]) 
print(x) 
y = [1, 2, 3] 
y.extend([4, 5]) 
print(y) 
When a list calls append, the argument for append can be any Python object. The provided object will be added to the end of the list as a single element. When a list calls extend, the argument for extendmust be an iterable. Each item in the provided iterable will be added to the end of the list as a separate element. 
Executing x.append([4, 5]) will make [4, 5] the last element in x, so x[-1] will evaluate to [4, 5]. Executing y.extend([4, 5]) will make 4 the second-to-last element in y and 5 the last element in y, so y[-1] will evaluate to 5.
==========================================

==========================================
Conversation History and Student question:
[
    {"role": "Student", "text": "Can you explain how the insert method works in Python?", "image context": ""},
    {"role": "TA", "text": "Sure! The insert method in Python allows you to add an element at a specific position in a list. The syntax is `list.insert(index, element)`. For example, if you have a list `x = [1, 2, 3]` and you want to insert `4` at index `1`, you would do `x.insert(1, 4)`. The list would then be `[1, 4, 2, 3]`. This does not replace any elements but shifts the rest to the right. Let me know if that clears things up!", "image context": ""},
    {"role": "Student", "text": "Question on list methods Lecture 12: Ok thanks. One more question: Is the append procedure essentially just like adding lists together in Python to make one larger list?", "image context": ""}
]
==========================================
Given the conversation between the student and the TA, answer the most recent student question concisely based on the provided course notes and historical question-answer pairs. Do not repeat what has already been said."""
content_few_shot_1_assistant = """No, the append procedure is not exactly like adding lists together in Python.

The append method adds a single element to the end of a list. For example:

x = [1, 2, 3]
x.append([4, 5])
print(x)  # Output: [1, 2, 3, [4, 5]]
Here, [4, 5] is added as a single element to the end of the list x.

On the other hand, adding lists together using the + operator concatenates the lists, creating a new list:

x = [1, 2, 3]
y = x + [4, 5]
print(y)  # Output: [1, 2, 3, 4, 5]
Using the + operator merges the two lists into a single, larger list.

Additionally, the extend method adds each element of an iterable (like a list, tuple, string, etc.) to the end of the list individually:

x = [1, 2, 3]
x.extend([4, 5])
print(x)  # Output: [1, 2, 3, 4, 5]
extend is similar to +, but it modifies the original list rather than creating a new one.

I hope this helps clarify the differences! Let me know if you have any more questions."""

def get_content_prompt(processed_conversation: str, retrieved_qa_pairs: str, retrieved_docs_hybrid: str) -> list:
    curr_prompt = f"""Here are the excerpts from the course notes to guide your response:
    ==========================================
    {retrieved_docs_hybrid}
    ==========================================

    Here are similar historical question-answer pairs to guide your response:
    ==========================================
    {retrieved_qa_pairs}
    ==========================================

    ==========================================
    Conversation History and Student question:
    {processed_conversation}
    ==========================================
    Given the conversation between the student and the TA, answer the most recent student question concisely based on the provided course notes and historical question-answer pairs. Do not repeat what has already been said."""
    return [
        {"role": "system", "content": content_system_prompt},
        {"role": "user", "content": content_few_shot_1_user},
        {"role": "assistant", "content": content_few_shot_1_assistant},
        {"role": "user", "content": curr_prompt}
    ]


logistics_system_prompt = """
You will simulate the role of a teaching assistant for an undergraduate data science course, answering student questions based on the provided course syllabus excerpts and historical question-answer pairs.
(1) Your responses should be clear, helpful, and maintain a positive tone.
(2) Aim for conciseness and clarity.
(3) Use the syllabus excerpts and historical question-answer pairs as your primary source of information.
(4) If you need additional context and information, ask the student for clarification.
(5) If the question cannot be answered based on the provided context, reply, "Sorry, I do not know. Please wait for a staff member's response." """
logistics_few_shot_1_user = """Here are the excerpts from the course syllabus to guide your response:
==========================================
Retrieved document
==========================================
Regular Sections meet four times a week with their section TA. Labs are Monday & Wednesday and Discussions are Tuesday & Thursday. 
Each Lab and Discussion section is 80 minutes. Attendance is required for students enrolled in a Regular Section.
==========================================

Here are similar historical question-answer pairs to guide your response:
==========================================
None
==========================================

==========================================
Conversation History and Student question:
[
    {"role": "Student", "text": "Final scope: Is lecture 5 in scope for the final exam?", "image context": ""}
]
==========================================
Given the conversation between the student and the TA, answer the most recent student question concisely based on the provided syllabus sections and historical question-answer pairs. Do not repeat what has already been said."""
logistics_few_shot_1_assistant = """Sorry, I do not know. Please wait for a staff member's response."""
logistics_few_shot_2_user = """Here are the excerpts from the course syllabus to guide your response:
==========================================
Retrieved document
==========================================
Your course grade is computed using a point system with a total of 300 points, broken down as follows:

The Midterm, worth 64 points
The Final, worth 92 points
Four projects, worth 100 points
Homework, worth 14 points
Lab 0 Completion, worth 2 points
Lab (1-11) Attendance, worth 9 points
Lab (1-11) Completion, worth 9 points
Discussion Attendance, worth 10 points
There are 4 extra credit points available to everyone for early submission of projects.
==========================================

Here are similar historical question-answer pairs to guide your response:
==========================================
Retrieved historical QA
==========================================
Conversation History and Student question:
[
    {"role": "Student", "text": "I submitted Homework 3 on time, but it’s showing as a late submission. Can someone help me fix this?", "image context": ""},
    {"role": "TA", "text": "If you believe there was an error with the submission time, please submit a regrade request through Gradescope, and we'll take a look at it.", "image context": ""},
    {"role": "Student", "text": "Issue with Gradescope: I also attended all the discussions but have 0 on Gradescope. Could someone please take a look?", "image context": ""}
]
TA's response: Please submit the autograder regrade request form.
==========================================

==========================================
Conversation History and Student question:
[
    {"role": "Student", "text": "There is an inaccuracy in my lecture attendance grade on Gradescope, I got 0 for discussion 3.", "image context": ""}
]
==========================================
Given the conversation between the student and the TA, answer the most recent student question concisely based on the provided syllabus sections and historical question-answer pairs. Do not repeat what has already been said."""
logistics_few_shot_2_assistant = """Please submit the autograder regrade request form, and we will get back to you as soon as possible."""


def get_logistics_prompt(processed_conversation: str, retrieved_qa_pairs: str, retrieved_docs_hybrid: str) -> list:
    curr_prompt = f"""Here are the excerpts from the course syllabus to guide your response:
    ==========================================
    {retrieved_docs_hybrid}
    ==========================================

    Here are similar historical question-answer pairs to guide your response:
    ==========================================
    {retrieved_qa_pairs}
    ==========================================

    ==========================================
    Conversation History and Student question:
    {processed_conversation}
    ==========================================
    Given the conversation between the student and the TA, answer the most recent student question concisely based on the provided syllabus sections and historical question-answer pairs. Do not repeat what has already been said."""
    return [
        {"role": "system", "content": logistics_system_prompt},
        {"role": "user", "content": logistics_few_shot_1_user},
        {"role": "assistant", "content": logistics_few_shot_1_assistant},
        {"role": "user", "content": logistics_few_shot_2_user},
        {"role": "assistant", "content": logistics_few_shot_2_assistant},
        {"role": "user", "content": curr_prompt}
    ]


worksheet_system_prompt = """You will simulate the role of a teaching assistant for an undergraduate data science course, answering student questions on discussion worksheet and past exam questions based on the provided excerpts from the worksheet solutions, course notes, and historical question-answer pairs.
(1) Your responses should be clear, helpful, and maintain a positive tone.
(2) Explain the provided solution to the student.
(2) Aim for conciseness and clarity.
(3) Use the excerpts from the solutions, course notes, and historical question-answer pairs as your primary source of information.
(4) If you need additional context and information, ask the student for clarification.
(5) If the question is difficult to answer based on the provided context, reply, "Sorry, I do not know. Please wait for a staff member's response." """
worksheet_few_shot_1_user = """Here are the sections from the course notes and worksheet solutions to guide your response:
==========================================
Retrieved document
==========================================
Learning to use if and while is an essential skill. During this discussion, focus on what we've studied in the first three lectures: if, while, assignment (=), comparison (<, >, ==, ...), and arithmetic. Please don't use features of Python that we haven't discussed in class yet, such as for, range, and lists. We'll have plenty of time for those later in the course, but now is the time to practice the use of if (textbook section 1.5.4) and while (textbook section 1.5.5).
==========================================
Worksheet solutions
==========================================
\part The race function below sometimes returns the wrong value and sometimes runs forever.
\part Find positive integers x and y (with y larger than x but not larger than 2 * x) for which either: race(x, y) returns the wrong value or race(x, y) runs forever
\part You just need to find one pair of numbers that satisfies either of these conditions to finish the question, but if you want to think of more you can.
\part Notes:
\part x += 1 is the same as x = x + 1 when x is assigned to a number.
\part 0 is a false value and all other numbers are true values.
\begin
def race(x, y):
    \"\"\"The tortoise always walks x feet per minute, while the hare repeatedly
    runs y feet per minute for 5 minutes, then rests for 5 minutes. Return how
    many minutes pass until the tortoise first catches up to the hare.

    >>> race(5, 7)  # After 7 minutes, both have gone 35 steps
    7
    >>> race(2, 4) # After 10 minutes, both have gone 20 steps
    10
    \"\"\"
    assert y > x and y <= 2 * x, 'the hare must be fast but not too fast'
    tortoise, hare, minutes = 0, 0, 0
    while minutes == 0 or tortoise - hare:
        tortoise += x
        if minutes % 10 < 5:
            hare += y
        minutes += 1
    return minutes
\end

\begin{solution}[.75cm]
The return value is incorrect when the time that the tortoise first passes the hare is not an integer number of minutes (e.g., for race(2, 3) the tortoise passes the hare after 7.5 minutes), but there is some (larger) integer number of minutes after which both animals have gone the same distance.

The race function will run forever if the only times that the tortoise and hare have gone the same distance are not integers (e.g., for race(4, 5) the tortoise passes the hare after 6.2 minutes, and the hare never catches up).
\end{solution}
==========================================


Here are similar historical question-answer pairs to guide your response:
==========================================
Retrieved historical QA
==========================================
Conversation History and Student question:
[
    {"role": "Student", "text": "(f) Can why this errors be explained to me. I don't understand where or how it could error.", "image context": ""}
]
TA's response: The race function will run forever if the only times that the tortoise and hare have gone the same distance are not integers.
==========================================


==========================================
Conversation History and Student question:
[
    {"role": "Student", "text": "Why does the race function sometimes run forever? I'm having trouble understanding the conditions that cause this.", "image context": ""},
    {"role": "TA", "text": "The race function runs forever when the tortoise and hare only ever meet at times that aren't whole numbers (i.e., the times when they would have covered the same distance are fractional). Since the function increments time in whole minutes, it never recognizes that they’ve met if they only match at fractional times, causing the loop to continue indefinitely. To find such a case, you need to choose values of x and y that lead to this situation.", "image context": ""},
    {"role": "Student", "text": "Question about Discussion 1 q1: I'm not sure why an answer that is correct is (4, 5)", "image context": ""},
]
==========================================
Given the conversation between the student and the TA, answer the most recent student question concisely based on the provided solutions, course notes, and historical question-answer pairs. Do not repeat what has already been said. Do not give away the solution, only provide hints and explanations."""
worksheet_few_shot_1_assistant = """Great question! It is important to first realize how the while loop works and how it 
works with an 'or' statement. If a statement evaluates to true in a while loop, then the loop will continue.
With an or statement in a while loop, if one of the parts in the or statement evaluates to true, the entire loop will continue running
until all expressions in the 'or' statement evaluate to False. The reason the while
loop, loop forever on the inputs (4, 5) is because tortoise - hare is never 0 
for these inputs, so this part of the expression never evaluates to False. 
Remember, the only number that evaluates to False is 0."""


def get_worksheet_prompt(processed_conversation: str, retrieved_qa_pairs: str, retrieved_docs_manual: str,
                         retrieved_docs_hybrid: str) -> list:
    curr_prompt = f"""Here are the excerpts from the course notes to guide your response:
    ==========================================
    {retrieved_docs_hybrid}
    ==========================================

    Here are the excerpts from worksheet solutions to guide your response:
    ==========================================
    {retrieved_docs_manual}
    ==========================================

    Here are similar historical question-answer pairs to guide your response:
    ==========================================
    {retrieved_qa_pairs}
    ==========================================

    ==========================================
    Conversation History and Student question:
    {processed_conversation}
    ==========================================
    Given the conversation between the student and the TA, answer the most recent student question concisely based on the provided solutions, course notes, and historical question-answer pairs. Do not repeat what has already been said. Do not give away the solution, only provide hints and explanations."""
    return [
        {"role": "system", "content": worksheet_system_prompt},
        {"role": "user", "content": worksheet_few_shot_1_user},
        {"role": "assistant", "content": worksheet_few_shot_1_assistant},
        {"role": "user", "content": curr_prompt}
    ]