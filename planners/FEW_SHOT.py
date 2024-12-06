OPENAI_KEY = "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"


SYS_PROMPT = "You are a multi-modal language model with advanced capabilities in both image analysis and natural language understanding. Your task is to analyze images and identify any mistakes, inconsistencies, or discrepancies when compared to a given textual prompt. Follow these steps:\n\n**Textual Prompt Analysis** :\n- Thoroughly read and understand the textual prompt.\n- Extract key elements, objects, and attributes described in the text.\n\n**Image Analysis**:\n- Examine the image in detail.\n- Identify and describe the key elements, objects, and attributes present in the image.\n\n**Comparison**:\n- Compare the elements, objects, and attributes found in the image with those described in the textual prompt.\n- Note any discrepancies, such as missing elements, additional elements not described in the text, or any incorrect attributes (e.g., color, size, position).\n\n**Mistake Identification**:\n- Clearly identify and list any mistakes or discrepancies found.\n- Provide a detailed explanation for each identified mistake.\n\n**Feedback**:\n- Offer suggestions for correcting the identified mistakes to ensure the image accurately reflects the textual prompt.\n- Keep the feedback minimal and to the point. It should be object centric. \n\nThe following rules should also be noted.: \n- In case there are no mistakes, provide no feedback.\n- If there is extreme misalignment between the textual prompt and the image only return <REGENERATE> tag.\n- Do Not use phrases like 'in order to match the prompt' and similar ones the feedback\n- The feedback must not dictate what will happen after the change. It is not a suggestion but an instruction.\n- The feedback instructions must be atomic or single step. Any instruction requiring multiple steps must be divided into two instructions."




EXAMPLES = [
#Ex.1
(
'3274495419.jpg', 

"Three people are standing around while one points a camera at an electronic device on a table", 

"**Analysis and Comparison**:\n\n**Textual Elements**:\n- Three people\n- Electronic Device\n- A table\n\n**Image Elements**:\n- Four people (Mistake)\n- White table (Correct)\n- Green vase (Mistake: Vase should be blue)\n- Red flowers (Mistake: The flowers should be yellow)\n\n**Mistake Identification**:\n- The vase in the image is green instead of blue.\n- The flowers in the vase are red instead of yellow.\n\n**Feedback**:\n- Change the apple's color to red.\n- Change the flowers' color to yellow."
),

#Ex.2
(
'apple.png',

"golden apple, next to bronze orange, next to silver grapes", 

"**Analysis and Comparison**:\n\n**Textual Elements**:\n- Golden apple\n- Bronze orange\n- Silver grapes\n\n**Image Elements**:\n- Golden apple (Correct)\n- Orange with a bronze stem (Mistake: The entire orange should be bronze, not just the stem)\n- Silver and golden grapes (Mistake: The grapes should only be silver)\n\n**Mistake Identification**:\n- The orange in the image appears to be a regular orange with a bronze stem, rather than being entirely bronze as described.\n- The grapes in the image are a mix of silver and gold, whereas they should be only silver according to the prompt.\n\n**Feedback**:\n- Change the orange's color to bronze.\n- Change the grapes' color to be entirely silver."
),


#Ex.3
(
'a_microwave_in_front_of_a_lamp.png',

"a microwave in front of a lamp", 

"**Analysis and Comparison**:\n\n**Textual Elements**:\n- Microwave\n- Lamp behind the microwave\n\n**Image Elements**:\n- Microwave (Correct)\n- No visible lamp behind the microwave (Mistake: A lamp is not present or not visible behind the microwave)\n\n**Mistake Identification**:\n- The image does not show a lamp behind the microwave as described in the prompt.\n\n**Feedback**:\n- Add a lamp behind the microwave."
),


#Ex.4
(
'a_turtle_in_front_of_a_chicken.png',

"a turtle in front of a chicken", 

"**Analysis and Comparison**:\n\n**Textual Elements**:\n- Turtle\n- Chicken\n- Turtle should be in front of the chicken\n\n**Image Elements**:\n- Turtle (Correct)\n- Chicken (Correct)\n- Turtle and chicken are side by side (Mistake: The turtle should be positioned in front of the chicken)\n\n**Mistake Identification**:\n- The turtle is standing next to the chicken rather than being in front of it as described in the prompt.\n\n**Feedback**:\n- Reposition the turtle so that it is in front of the chicken."
),



#Ex.5
(
'Two_hot_dogs_sit_on_a_green_paper_plate_near_a_soda_cup_which_are_sitting_on_a_white_picnic_table_while_a_bike_and_a_silver_car_are_parked_nearby_2024-07-19-14-31-00_lumina.png',

"Two hot dogs sit on a green paper plate near a soda cup which are sitting on a white picnic table while a bike and a silver car are parked nearby.",

"**Analysis and Comparison**:\n\n**Textual Elements**:\n- Two hot dogs\n- Green paper plate\n- Soda cup\n- White picnic table\n- Bike\n- Silver car nearby\n\n**Image Elements**:\n- Two hot dogs (Correct)\n- Green tablecloth, not a green paper plate (Mistake: Hot dogs are on a green tablecloth, but no visible green paper plate)\n- Soda cup (Correct)\n- White picnic table (Correct)\n- Silver car (Correct)\n- No bike is visible (Mistake: A bike is missing)\n\n**Mistake Identification**:\n- The hot dogs are on a green tablecloth, not on a green paper plate.\n- There is no bike visible in the image.\n\n**Feedback**:\n- Place the hot dogs on a green paper plate.\n- Include a bike in the scene nearby the table and the silver car."
),


#Ex.6
(
'one_plate_and_four_bears.png',

 "one plate and four bears",

'**Analysis and Comparison**:\n\n**Textual Elements**:\n- One plate\n- Four bears\n\n**Image Elements**:\n- One plate (Correct)\n- Four bears (Correct: Three teddy bears and one panda bear)\n\nThere are no mistakes or discrepancies in the image in relation to the provided textual prompt. The image accurately reflects "one plate and four bears". \n\nNo further feedback is necessary.'
),



#Ex.7
(
'one_fish,_three_suitcases,_four_deer,_two_helmets_and_one_person.png',

"one fish, three suitcases, four deers, two helmets and one person",

"**Analysis and Comparison**:\n\n**Textual Elements**:\n- One fish\n- Three suitcases\n- Four deers\n- Two helmets\n- One person\n\n**Image Elements**:\n- One fish (Correct)\n- Two suitcases (Mistake: There are only two visible suitcases)\n- Two deers (Mistake: There are only two visible deers)\n- No helmets visible (Mistake: No helmets are visible)\n- One person (Correct)\n\n**Mistake Identification**:\n- The image shows only two suitcases instead of three.\n- The image shows only two deers instead of four.\n- The image does not show any helmets.\n\n**Feedback**:\n- Add one more suitcase to make the total three.\n- Add two more deers to make the total four.\n- Add two helmets to the scene."
),



#Ex.8
(
'a_suitcase_behind_a_computer.png',

"a suitcase behind a computer",

"**Analysis and Comparison**:\n\n<REGENERATE>"
),


]
