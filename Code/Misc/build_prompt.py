#Use OpenAI/Langchain

def build_examples_prompt(concern_type, concerns_list, examples):
    prompt_builder = f"Give me at least 1000 examples. Augment data for NLP that indicate that there is a {concern_type} concern. Anything that the instructor should look at such as {', '.join(concerns_list)}\nhere are examples\n"
    example_template = "{}\n"


    for example in examples:
        prompt_builder += example_template.format(example)

    return prompt_builder

# Here are the examples I used
'''
time_management_concerns = ["time management"]
examples_time = [
        "I struggle to keep track of all deadlines",
        "I have a tough time making all the deadlines on time",
        "Too much to keep track of",
        "I don't know how to cram all this knowledge in the limited time I have"
]

examples_other = [
    "I have cancer",
    "I am depressed",
    "I need accomodations for my ADHD",
    "I know people in this class who are cheating",
    "I have a question on how to access the textbook",
    "I need to discuss my 504 plan",
    "This class is giving me anxiety",
    "The tests give me anxiety and make me nervous, I have test anxiety",
    "I need help for my disability",
    "I need accomodations for my wheelechair disability",
    "Would it be okay to come in during office hours to ask for help with things that were covered in pre-requisite courses?",
    "Is there a list of all the vocabulary terms that are useful for this course?",
    "I would like to know if we are going to use any new notations on homework.",
    "Is it ok to work with other people to finish homework?",
    "I broke my hand and need help writing",
    "my mother has covid",
    "I have Covid",
    "I am feeling awful, nothing seems to work and I hate this course",
    "I want to contact the dean of students",
    "How do I set up a meeting with you?",
    "How do I get one on one tutoring?"

]

other_concerns = ["mental health issues", "OSI (office of student integrity)", "Dean of Students", "Disability", "Unstable", "Anxiety", "Accommodation", "ADH", "Disorder", "Depression", "Travel Plans", "504 plan", "any health concern", "any question"]
lm_concerns =["Difficulty Learning material", "Exam Stress", "Reviewing Pre Requisite material for calculus or some other previous class", "Any struggles with learning the material itself", "Finding material difficult"]
examples_lm = [
    "That I forget all my knowledge from calc BC",
    "My only concern about this course is that I will have an issue learning one of the fundamental items early in the course, and as a result of that, will struggle later on when building upon that foundational knowledge. ",
    "The material is really tough and its just not getting to my brain",
    "I'm struggling to learn, I just don't get it",
    "I don't rememeber calculus at all",
    "I don't know how to study for the exams, its so tough to review all that information :(",
    "I found topics related to series really tough last year and I feel like it might be tough to pick up on its content"


]
time_manage_prompt = build_examples_prompt("time management", time_management_concerns, examples_time)
other_prompt = build_examples_prompt("some other important concern", other_concerns, examples_other)
lm_prompt = build_examples_prompt("learning management concern", lm_concerns, examples_lm)
print("Time Management Prompt:")
print(time_manage_prompt)

print("\nOther Concern Prompt:")
print(other_prompt)

print("\nLearning management Concern Prompt:")
print(lm_prompt)
'''