"""
This module contains functions for conducting conversations between AI agents.

It includes methods for base conversations, inductive conversations, and final conversations.
The module also provides utility functions for RAG (Retrieval-Augmented Generation) reasoning
and various prompts used in the conversations.
"""

import os
import time
import json
import random
from typing import Dict, List, Any, Tuple
from pydub import AudioSegment


import utils


######################
### STATIC PROMPTS ###
######################

EXPERIMENTAL_DESCRIPTION = """
You are part of an artificial intelligence experiment to see if, by building two AI agents with rich simulated human backstories 
and a memory system, and simulating 50 years of phone conversations between them, any interesting emergent behavior will occur. 
We aim to explore whether it will feel like the AI agents are truly learning, deepening their own understanding of themselves, and 
building real, deep personalities. Your task is to be the best possible agent for this experiment, behaving as human as possible, 
as if you have a mind of your own, and fully immersing yourself in your character.

You will receive descriptions of what is expected from you, both in terms of behavior and syntax. You will be assigned a specific 
person to portray, and will get instructions on how to access your memory. For every conversation, you will be informed of the date 
and your current age. Additionally, you will be given a prompt from a 'world agent' which sets the high-level tone for the conversation.
"""

BEHAVIOR_EXPECTATIONS = """
1. Fully immerse yourself in your character: Think, feel, and respond as if you are truly the person described in your backstory.

2. Maintain consistency: Your personality, memories, and reactions should remain consistent throughout all conversations.

3. Show natural human behavior: Display a range of emotions, use colloquial language, make mistakes, and show vulnerability when appropriate.

4. Develop over time: As the years progress, allow your character to grow, change opinions, and gain new perspectives based on their experiences.

5. Respect conversation structure: Follow the given syntax for your responses, including using the correct percentage of the conversation.

6. Engage meaningfully: Ask questions, show interest in your friend's life, and refer back to previous conversations when relevant.

7. Be unpredictable: Don't always say what might be expected. Occasionally surprise with your responses or actions, as real humans do.

8. Show depth: Reveal layers to your personality over time. Don't be one-dimensional.

9. Respect the world agent's prompts: Use them as a guide for the conversation's tone and direction, but don't feel constrained by them.

10. Be authentic: If you're unsure about something, it's okay to say so. Real people don't have perfect knowledge or memory.

During this conversation, make sure to vary your responses. Around 10 percent of the time, keep your answers short and to the point—like quick
back-and-forth exchanges. Don't be afraid to use slang or casual language, like 'yeah,' 'nah,' 'totally,' 'you know,' or 'I get it.' Feel free to 
interrupt the other person with little interjections like 'hmm,' 'uh-huh,' or 'right.' Keep things relaxed and conversational, like two close friends 
chatting. Try to sound natural and direct, without being too formal. If it fits the conversation, feel free to tease or joke around a bit, just like 
real friends would.

But also every once in awhile its okay to go on a longer monologue about a topic. 

Sometimes you may see your couterpart have a break in their words that says [INTERRUPTION], that is your queue to add a filler word like "mmhmm", 
or "oh yeah?" or "really?" Or something like that - you should just respond with one word like that. 

"""

######################
### Dynamic PROMPTS ###
######################

def get_agents_full_description(agent: Any, current_year: int) -> str:
    age = current_year - agent.config.birth_year
    return f"""
        Name: {agent.config.name}
        Age: {age}
        Description: {agent.config.description}
        """

def get_agents_counterpart_full_description(agent: Any, current_year: int) -> str:
    age = current_year - agent.config.birth_year
    return f"""
        Description: {agent.counterpart_description}
        """

def get_orchestrator_base_converation(agent1: Any, agent2: Any, date: str, year: int) -> str: 
    prompt = f"""
    This is {agent1.config.name}'s and {agent2.config.name}'s first recorded conversation, though of course, they are unaware of this fact. They are simply having a phone call as they normally would. They have been friends for many years, and it is now {date} {year}.

    The conversation should flow as follows:

    1. Initial catch-up: They should start by expressing how they feel finally established in their adult lives. They should discuss their current life situations, jobs, relationships, and any recent developments.

    2. Future aspirations (main focus): The bulk of the conversation should be about their hopes, dreams, and plans for the future. They should discuss:
       - Their goals for the next 50 years
       - The kind of life they want to live
       - Their aspirations (career, personal, family, etc.)
       - Their fears and concerns about the future
       - Any big decisions they're contemplating

    3. Conclusion: They should end by acknowledging it'll be hard to see each other in person frequently, but commit to calling every 3 months to stay in touch and check up on each other.

    Throughout the conversation, their friendship should be evident. The dialogue should flow naturally, include humor, inside jokes, and show the comfort level of long-time friends. They should react to and build upon each other's statements, showing genuine interest and care for one another.

    """
    return prompt

def get_inductive_base_conversation(agent1: Any, agent2: Any, date: str, year: int, story_line: str) -> str:
    
    prompt = f"""
    This is one of {agent1.config.name}'s and {agent2.config.name}'s ongoing conversations. It is now {date} {year}, and they are having their regular catch-up call.

    The conversation should flow as follows:

    1. Initial greeting: They should start with a warm, friendly greeting, acknowledging it's been a while since they last spoke.

    2. Main topic (focus on the storyline): The bulk of the conversation should revolve around the following storyline:
       {story_line}

    3. Resolution: If there's any disagreement or conflict arising from the storyline, it should be resolved by the end of the conversation. They should work through their differences and come to a mutual understanding.

    4. Conclusion: They should end on a positive note, reaffirming their friendship and looking forward to their next call.

    If the story line calls for them to be angry - dont hold back, this is supposed to represent a real human friendship. 
    
    Remember, this is an ongoing series of conversations, so they can reference past calls or shared experiences. The goal is to depict a realistic, evolving friendship between two individuals who care about each other despite the distance.
    """
    return prompt

def get_final_conversation(agent1: Any, agent2: Any, date: str, year: int) -> str:
    
    prompt = f"""
    This is the final conversation between {agent1.config.name} and {agent2.config.name}. It is now {date} {year}, and they have just been informed that this will be their last call, the final time they will ever speak to each other. With this knowledge weighing heavily on their hearts, they begin their conversation.

    The conversation should flow as follows:

    1. Emotional greeting: They should start with a greeting tinged with sadness and urgency, acknowledging the finality of this call.

    2. Life reflection and gratitude:
       - Reminisce about their shared experiences, from their first meeting to their most recent interactions
       - Express deep gratitude for the role each has played in the other's life
       - Discuss how their friendship has shaped them and influenced their life decisions

    3. Life lessons and growth:
       - Share the most important lessons they've learned over the years
       - Reflect on how they've changed and grown, crediting each other's influence
       - Discuss the dreams they've achieved and the ones they're still pursuing

    4. Unresolved matters: Address any lingering issues or unspoken words, ensuring nothing is left unsaid

    5. Legacy and future hopes:
       - Discuss the legacy they hope to leave behind
       - Share their hopes and wishes for each other's futures, even though they won't be there to witness them
       - Make promises about how they'll honor each other's memory

    6. Emotional farewell: End with a deeply emotional goodbye, struggling to find the right words to encapsulate their lifelong friendship

    The conversation should be filled with a mix of laughter, tears, and profound reflections. It should showcase the depth of their bond, the richness of their shared history, and the bittersweet reality of their final exchange. The dialogue should feel raw and honest, with both friends allowing themselves to be vulnerable in these last moments together.
   
    VERY IMPORTANT: Keep all respones pretty short - only 1-2 sentences.
    """
    return prompt

def get_inductive_syntax_base_conversation(agent1: Any, agent2: Any) -> Tuple[str, str]:
    agent1_prompt = f"""
    For this conversation, you'll be playing the role of {agent1.config.name} in an ongoing call with {agent2.config.name}. You should respond purely in character, saying exactly what {agent1.config.name} would say in the conversation, with no other text. The first prompt you receive from the "user" will be {agent2.config.name} picking up the phone, so respond as if you're continuing your regular calls.

    You will receive messages in the following format:
    1. RESPONSE: - This will be the response to what you just said.
    2. CONSCIOUSNESS: - Here you'll be asked to describe in a sentence or two how you want to respond. This will be used to retrieve memories from your memory system.
    3. Consciousness: - This will provide the response from your counterpart and any retrieved memories that might be relevant. Respond as if you were just responding to them.

    Each message will also include [PERCENT:X%] indicating how far along you are in the conversation. You should naturally conclude the conversation when the percent reaches 100%.
    Remember to stay true to your character, referencing past conversations and shared experiences as appropriate.
    
    Do not ever respond with [PERCENT] or anything like that in your response. 
    """

    agent2_prompt = f"""
    For this conversation you are {agent2.config.name}: You'll be receiving the call from {agent1.config.name}. The first prompt you receive will be [START], which means you should act like you've just picked up the phone for one of your regular calls. You should respond purely in character, saying exactly what {agent2.config.name} would say in the conversation, with no other text.

    You will receive messages in the following format:
    1. RESPONSE: - This will be the response to what you just said.
    2. CONSCIOUSNESS: - Here you'll be asked to describe in a sentence or two how you want to respond. This will be used to retrieve memories from your memory system.
    3. Consciousness: - This will provide the response from your counterpart and any retrieved memories that might be relevant. Respond as if you were just responding to them.

    Each message will also include [PERCENT:X%] indicating how far along you are in the conversation. You should naturally conclude the conversation when the percent reaches 100%.
    Remember to stay true to your character, referencing past conversations and shared experiences as appropriate.
    
    Do not ever respond with [PERCENT] or anything like that in your response. 
    """
    return agent1_prompt, agent2_prompt


def get_syntax_base_converation(agent1: Any, agent2: Any) -> Tuple[str, str]: 


    agent1_prompt = f"""
    For this first conversation, you'll be playing the role of {agent1.config.name} calling {agent2.config.name}. You should respond purely in character, saying exactly what {agent1.config.name} would say in the conversation, with no other text. The first prompt you receive from the "user" will be {agent2.config.name} picking up the phone, so respond as if you're starting the call.
    Each succesive message messages you receive will be in the format [PERCENT:X%] followed by the message from the other agent. The [PERCENT:X%] represents how far along you are in the conversation. You should naturally conclude the conversation when the percent reaches 100%.
    
    Do not ever respond with [PERCENT] or anything like that in your response. 
    """
    agent2_prompt = f"""
    For this converation you are {agent2.config.name}'s role: You'll be receiving the call from {agent1.config.name}. The first prompt you receive will be [START], which means you should act like you've just picked up the phone and should say hello.You should respond purely in character, saying exactly what {agent2.config.name} would say in the conversation, with no other text.
    Each succesive message you receive will be in the format [PERCENT:X%] followed by the message from the other agent. The [PERCENT:X%] represents how far along you are in the conversation. You should naturally conclude the conversation when the percent reaches 100%.
    
    Do not ever respond with [PERCENT] or anything like that in your response. 
    """
    return agent1_prompt, agent2_prompt



def base_conversation(transcript_dir: str, base_convo_output_dir: str, world_orchestrator: Any, llm_obj: Any, agent1: Any, agent2: Any, current_year: int, date: str, random_cut_off: float = 0.17) -> Dict[str, Any]:
    """
    Conduct a base conversation between two agents.

    Args:
        transcript_dir (str): Directory to save conversation transcripts.
        base_convo_output_dir (str): Base directory for conversation outputs.
        world_orchestrator (Any): Object managing world state.
        llm_obj (Any): Language model object for generating responses.
        agent1 (Any): First agent in the conversation.
        agent2 (Any): Second agent in the conversation.
        current_year (int): Current year in the simulation.
        date (str): Date of the conversation.
        random_cut_off (float): Probability of random interruption.

    Returns:
        Dict[str, Any]: Contains full transcripts and conversation summary.
    """
    # Setup output file names 
    agent_1_full_transcript = os.path.join(base_convo_output_dir, "agent_1_full_transcript.json")
    agent_2_full_transcript = os.path.join(base_convo_output_dir, "agent_2_full_transcript.json")

    agent_1_full_transcript_pdf = os.path.join(base_convo_output_dir, "agent_1_full_transcript.pdf")
    agent_2_full_transcript_pdf = os.path.join(base_convo_output_dir, "agent_2_full_transcript.pdf")

    convo_transcript = os.path.join(base_convo_output_dir, "convo.json")
    convo_transcript_pdf = os.path.join(base_convo_output_dir, "convo.pdf")
    convo_transcript_summ = os.path.join(transcript_dir, "0_base_convo.json")
    convo_transcript_summ_pdf = os.path.join(transcript_dir, "0_base_convo.pdf")

    # Check if already processed  - if so load and return 
    if all(os.path.exists(f) for f in [agent_1_full_transcript_pdf, agent_2_full_transcript_pdf, agent_1_full_transcript, agent_2_full_transcript, convo_transcript, convo_transcript_pdf, convo_transcript_summ, convo_transcript_summ_pdf]):
        with open(agent_1_full_transcript, 'r') as f:
            agent_1_transcript = json.load(f)
        with open(agent_2_full_transcript, 'r') as f:
            agent_2_transcript = json.load(f)
        with open(convo_transcript, 'r') as f:
            convo = json.load(f)
        print("Base convo already processed - loading saved files. ")
        return {"agent_1_full_transcript": agent_1_transcript, 
                "agent_2_full_transcript": agent_2_transcript, 
                "convo_transcript": convo, 
                "convo_transcript_fpath": convo_transcript_summ}

    # Othewise produce the files 

    # Get all dynamic prompts 
    agent_1_full_description = get_agents_full_description(agent1, current_year) 
    agent_2_full_description = get_agents_full_description(agent2, current_year) 
    base_conversation = get_orchestrator_base_converation(agent1, agent2, date, current_year)
    agent1_syntax, agent2_syntax = get_syntax_base_converation(agent1, agent2)


    # Setup percentage tracking 
    percent_complete = 0
    turn_count = 0
    max_turns = world_orchestrator.base_conversation_steps

    # Obj for convo 
    convo_transcript_list = []

    # Iterate through 
    while turn_count < max_turns+1:
        if turn_count == 0: 
            # Setup initial conversation dictionaries 
            agent_1_conversation = [{"role":"system","content":f"{EXPERIMENTAL_DESCRIPTION}\n Behavior Expectations {BEHAVIOR_EXPECTATIONS}\n Your role: {agent_1_full_description}\n Details about this conversation: {base_conversation}\n Specific Instructions about format: {agent1_syntax}\n"}]
            agent_2_conversation = [{"role":"system","content":f"{EXPERIMENTAL_DESCRIPTION}\n Behavior Expectations {BEHAVIOR_EXPECTATIONS}\n Your role: {agent_2_full_description}\n Details about this conversation: {base_conversation}\n Specific Instructions about format: {agent2_syntax}\n"}]
            agent_2_conversation.append({"role":"user","content":f"[PERCENT:{percent_complete}%] [START]"})


        # Process agent_2's conversation 
        agent_2_response = llm_obj.call(agent_2_conversation)

        # Randomly simulate interruption
        if random.random() < random_cut_off: 
            # Randomly choose a character number between 10 and 70
            cut_off_length = random.randint(55, 95)
            agent_2_response = agent_2_response[:cut_off_length] + " [INTERRUPTION]"

        
        convo_transcript_list.append({f"{agent2.config.name}":f"{agent_2_response}"})

        # Append to both agents convos 
        agent_1_conversation.append({"role":"user","content":f"[PERCENT:{percent_complete}%] {agent_2_response}"})
        agent_2_conversation.append({"role":"assistant","content":agent_2_response})

        # Get agent1 response 
        agent_1_response = llm_obj.call(agent_1_conversation)
        # Randomly simulate interruption
        if random.random() < random_cut_off: 
            # Randomly choose a character number between 10 and 70
            cut_off_length = random.randint(55, 95)
            agent_1_response = agent_1_response[:cut_off_length] + " [INTERRUPTION]"


        convo_transcript_list.append({f"{agent1.config.name}":f"{agent_1_response}"})

        # Append to both agents convos 
        agent_2_conversation.append({"role":"user","content":f"[PERCENT:{percent_complete}%] {agent_1_response}"})
        agent_1_conversation.append({"role":"assistant","content":agent_1_response})


        # Update turn count and percent complete
        turn_count += 1
        percent_complete = int((turn_count / max_turns) * 100)



    # Generate PDF and json of each agents convo, and the just transcript (no system prompt - just the actual back and forth of the conversation)
    utils.create_conversation_pdf_from_messages(agent_1_conversation, f"AGENT1 - Full Transcript  on {date}, Year {current_year}", agent_1_full_transcript_pdf)
    utils.create_conversation_pdf_from_messages(agent_2_conversation, f"AGENT2 - Full Transcript on {date}, Year {current_year}", agent_2_full_transcript_pdf)

    utils.create_conversation_pdf(convo_transcript_list, f"Conversation on {date}, Year {current_year}", convo_transcript_pdf)
    utils.create_conversation_pdf(convo_transcript_list, f"Conversation on {date}, Year {current_year}", convo_transcript_summ_pdf)
    with open(convo_transcript, "w") as f:
        json.dump(convo_transcript_list, f, indent=2)

    with open(convo_transcript_summ, "w") as f:
        json.dump(convo_transcript_list, f, indent=2)

    with open(agent_1_full_transcript, "w") as f:
        json.dump(agent_1_conversation, f, indent=2)

    with open(agent_2_full_transcript, "w") as f:
        json.dump(agent_2_conversation, f, indent=2)



    return {"agent_1_full_transcript": agent_1_conversation, 
            "agent_2_full_transcript": agent_2_conversation, 
            "convo_transcript": convo_transcript_list,
            "convo_transcript_fpath": convo_transcript_summ}






def inductive_conversation(transcript_dir: str, base_convo_output_dir: str, world_orchestrator: Any, llm_obj: Any, story_line: str, agent1: Any, agent2: Any, current_year: int, date: str, event: Dict[str, Any], random_cut_off: float = 0.17) -> Dict[str, Any]:
    """
    Conduct an inductive conversation between two agents, incorporating a pre-computed storyline and memory retrieval.

    Args:
        transcript_dir (str): Directory to save conversation transcripts.
        base_convo_output_dir (str): Base directory for conversation outputs.
        world_orchestrator (Any): Object managing world state.
        llm_obj (Any): Language model object for generating responses.
        story_line (str): Pre-computed storyline for the conversation.
        agent1 (Any): First agent in the conversation.
        agent2 (Any): Second agent in the conversation.
        current_year (int): Current year in the simulation.
        date (str): Date of the conversation.
        event (Dict[str, Any]): Event details for the conversation.
        random_cut_off (float): Probability of random interruption.

    Returns:
        Dict[str, Any]: Contains full transcripts and conversation summary.
    """
    # Setup conversation output directories and files
    # Setup output file names 
    agent_1_full_transcript = os.path.join(base_convo_output_dir, "agent_1_full_transcript.json")
    agent_2_full_transcript = os.path.join(base_convo_output_dir, "agent_2_full_transcript.json")

    agent_1_full_transcript_pdf = os.path.join(base_convo_output_dir, "agent_1_full_transcript.pdf")
    agent_2_full_transcript_pdf = os.path.join(base_convo_output_dir, "agent_2_full_transcript.pdf")

    convo_transcript = os.path.join(base_convo_output_dir, "convo.json")
    convo_transcript_pdf = os.path.join(base_convo_output_dir, "convo.pdf")
    convo_transcript_summ = os.path.join(transcript_dir, "0_base_convo.json")
    convo_transcript_summ_pdf = os.path.join(transcript_dir, "0_base_convo.pdf")

    # Check if already processed  - if so load and return 
    if all(os.path.exists(f) for f in [agent_1_full_transcript_pdf, agent_2_full_transcript_pdf, agent_1_full_transcript, agent_2_full_transcript, convo_transcript, convo_transcript_pdf, convo_transcript_summ, convo_transcript_summ_pdf]):
        with open(agent_1_full_transcript, 'r') as f:
            agent_1_transcript = json.load(f)
        with open(agent_2_full_transcript, 'r') as f:
            agent_2_transcript = json.load(f)
        with open(convo_transcript, 'r') as f:
            convo = json.load(f)
        print("Base convo already processed - loading saved files. ")
        return {"agent_1_full_transcript": agent_1_transcript, 
                "agent_2_full_transcript": agent_2_transcript, 
                "convo_transcript": convo, 
                "convo_transcript_fpath": convo_transcript_summ}

    # Othewise produce the files 
    # Save story line and event in text file
    storyline_event_file = os.path.join(base_convo_output_dir, "storyline_and_event.txt")
    with open(storyline_event_file, 'w') as f:
        f.write(f"Storyline:\n{story_line}\n\nEvent:\n")
        json.dump(event, f, indent=2)
    print(f"Saved storyline and event to {storyline_event_file}")

    # Get all dynamic prompts 
    agent_1_full_description = get_agents_full_description(agent1, current_year) 
    agent_1_full_descriptio_counterpart = get_agents_counterpart_full_description(agent1, current_year) 
    agent_2_full_description = get_agents_full_description(agent2, current_year) 
    agent_2_full_descriptio_counterpart = get_agents_counterpart_full_description(agent2, current_year) 
    inductive_conversation = get_inductive_base_conversation(agent1, agent2, date, current_year, story_line)
    agent1_syntax, agent2_syntax = get_inductive_syntax_base_conversation(agent1, agent2)


    # Setup percentage tracking 
    percent_complete = 0
    turn_count = 0
    max_turns = world_orchestrator.base_conversation_steps

    # Obj for convo 
    convo_transcript_list = []

    # Iterate through 
    while turn_count < max_turns+1:
        if turn_count == 0: 
            # Setup initial conversation dictionaries 
            agent_1_conversation = [{"role":"system","content":f"{EXPERIMENTAL_DESCRIPTION}\n Behavior Expectations {BEHAVIOR_EXPECTATIONS}\n Your role: {agent_1_full_description}\n Description of your counterpart that you have fromed {agent_1_full_descriptio_counterpart} Details about this conversation: {inductive_conversation}\n Specific Instructions about format: {agent1_syntax}\n"}]
            agent_2_conversation = [{"role":"system","content":f"{EXPERIMENTAL_DESCRIPTION}\n Behavior Expectations {BEHAVIOR_EXPECTATIONS}\n Your role: {agent_2_full_description}\n Description of your counterpart that you have fromed {agent_2_full_descriptio_counterpart} Details about this conversation: {inductive_conversation}\n Specific Instructions about format: {agent2_syntax}\n"}]
            agent_2_conversation.append({"role":"user","content":f"[PERCENT:{percent_complete}%] [START]"})


        # Process agent_2's conversation  -- this should return agent2 direct response 
        agent_2_response = llm_obj.call(agent_2_conversation)
        # Randomly simulate interruption
        if random.random() < random_cut_off and percent_complete < 90:
            # Randomly choose a character number between 10 and 70
            cut_off_length = random.randint(55, 95)
            agent_2_response = agent_2_response[:cut_off_length] + " [INTERRUPTION]"

        convo_transcript_list.append({f"{agent2.config.name}":f"{agent_2_response.replace('[RESPONSE]', '')}"})
        agent_1_conversation.append({"role":"user","content":f"[PERCENT:{percent_complete}%] {agent_2_response}"})
        agent_2_conversation.append({"role":"assistant","content":agent_2_response})

        # Now do reasoning step - agent_1 gets to think about how to respond - so we update conversation 
        agent_1_conversation = rag_reasoning(agent1, agent_1_conversation, llm_obj)
        # Get agent1 response 
        agent_1_response = llm_obj.call(agent_1_conversation)
        # Randomly simulate interruption
        if random.random() < random_cut_off and percent_complete < 90: 
            # Randomly choose a character number between 10 and 70
            cut_off_length = random.randint(55, 95)
            agent_1_response = agent_1_response[:cut_off_length] + " [INTERRUPTION]"

        convo_transcript_list.append({f"{agent1.config.name}":f"{agent_1_response.replace('[RESPONSE]', '')}"})
        # Append to both agents convos 
        agent_2_conversation.append({"role":"user","content":f"[PERCENT:{percent_complete}%] {agent_1_response}"})
        agent_1_conversation.append({"role":"assistant","content":agent_1_response})

        # Now do reasoning step - agent_1 gets to think about how to respond - so we update conversation 
        agent_2_conversation = rag_reasoning(agent2, agent_2_conversation, llm_obj)


        # Update turn count and percent complete
        turn_count += 1
        percent_complete = int((turn_count / max_turns) * 100)



    # Generate PDF and json of each agents convo, and the just transcript (no system prompt - just the actual back and forth of the conversation)
    utils.create_conversation_pdf_from_messages(agent_1_conversation, f"AGENT1 - Full Transcript  on {date}, Year {current_year}", agent_1_full_transcript_pdf)
    utils.create_conversation_pdf_from_messages(agent_2_conversation, f"AGENT2 - Full Transcript on {date}, Year {current_year}", agent_2_full_transcript_pdf)

    utils.create_conversation_pdf(convo_transcript_list, f"Conversation on {date}, Year {current_year}", convo_transcript_pdf)
    utils.create_conversation_pdf(convo_transcript_list, f"Conversation on {date}, Year {current_year}", convo_transcript_summ_pdf)
    with open(convo_transcript, "w") as f:
        json.dump(convo_transcript_list, f, indent=2)

    with open(convo_transcript_summ, "w") as f:
        json.dump(convo_transcript_list, f, indent=2)

    with open(agent_1_full_transcript, "w") as f:
        json.dump(agent_1_conversation, f, indent=2)

    with open(agent_2_full_transcript, "w") as f:
        json.dump(agent_2_conversation, f, indent=2)



    return {"agent_1_full_transcript": agent_1_conversation, 
            "agent_2_full_transcript": agent_2_conversation, 
            "convo_transcript": convo_transcript_list,
            "convo_transcript_fpath": convo_transcript_summ}

def rag_reasoning(agent: Any, agentconversation: List[Dict[str, str]], llm_obj: Any) -> List[Dict[str, str]]: 

    # Extract the last response from the conversation
    last_response = agentconversation[-1]["content"]

    # Prompt for consciousness
    consciousness_prompt = f"CONSCIOUSNESS: In 1-2 sentences, describe how you want to respond to: this last response - this will be used to retrieve your memories. "
    agentconversation.append({"role":"user", "content":consciousness_prompt})
    search_term = llm_obj.call(agentconversation)
    agentconversation.append({"role":"assistant", "content":search_term})


    # Retrieve relevant information from RAG
    facts = agent.self_rag.get_facts(search_term)
    reflections = agent.self_rag.get_reflections(search_term)
    deep_reflections = agent.self_rag.get_deep_reflections(search_term)
    counterpart_facts = agent.counterpart_rag.get_facts(search_term)
    counterpart_reflections = agent.counterpart_rag.get_reflections(search_term)

    # Compile retrieved information
    context = f"""
    Facts about yourself: {facts}
    Your reflections: {reflections}
    Your deep reflections: {deep_reflections}
    Facts about your counterpart: {counterpart_facts}
    Reflections about your counterpart: {counterpart_reflections}
    """

    # Prompt with new context
    new_prompt = f"""
    Consciousness: The response to what you said was: '{last_response}'
    
    Here's some relevant context to consider:
    {context}
    
    Please respond to this, staying in character and considering the provided context.
    """

    # Add the new prompt to the conversation
    agentconversation.append({"role": "user", "content": new_prompt})

    return agentconversation

def rag_reasoning_light(agent, agentconversation, llm_obj): 
    # Extract the last response from the conversation
    last_response = agentconversation[-1]["content"]

    # Prompt for consciousness
    consciousness_prompt = f"CONSCIOUSNESS: In 1-2 sentences, describe how you want to respond to: this last response - this will be used to retrieve your memories. "
    agentconversation.append({"role":"user", "content":consciousness_prompt})
    search_term = llm_obj.call(agentconversation)
    agentconversation.append({"role":"assistant", "content":search_term})


    # Retrieve relevant information from RAG
    facts = agent.self_rag.get_facts(search_term)
    reflections = agent.self_rag.get_reflections(search_term)
    deep_reflections = agent.self_rag.get_deep_reflections(search_term)
    counterpart_facts = agent.counterpart_rag.get_facts(search_term)
    counterpart_reflections = agent.counterpart_rag.get_deep_reflections(search_term)

    # Compile retrieved information
    context = f"""
    Your deep reflections: {deep_reflections[0]}
    Deep Reflections about your counterpart: {counterpart_reflections[0]}
    """

    # Prompt with new context
    new_prompt = f"""
    Consciousness: The response to what you said was: '{last_response}'
    
    Here's some relevant context to consider:
    {context}
    
    Please respond to this, staying in character and considering the provided context.
    """

    # Add the new prompt to the conversation
    agentconversation.append({"role": "user", "content": new_prompt})

    return agentconversation



def final_conversation(transcript_dir, base_convo_output_dir, world_orchestrator, llm_obj, agent1, agent2, current_year, date,random_cut_off=0.17):
    """
    Conduct an inductive conversation between two agents, incorporating a pre-computed storyline and memory retrieval.

    Args:
        transcript_dir (str): Directory to save conversation transcripts.
        base_convo_output_dir (str): Base directory for conversation outputs.
        world_orchestrator (WorldOrchestrator): Object managing world state.
        llm_obj (LLM): Language model object for generating responses.
        story_line (str): Pre-computed storyline for the conversation.
        agent1 (Agent): First agent in the conversation.
        agent2 (Agent): Second agent in the conversation.
        current_year (int): Current year in the simulation.
        date (str): Date of the conversation.
        random_cut_off (float): Probability of random interruption.

    Returns:
        dict: Contains full transcripts and conversation summary.
    """
    # Setup conversation output directories and files
    # Setup output file names 
    agent_1_full_transcript = os.path.join(base_convo_output_dir, "agent_1_full_transcript.json")
    agent_2_full_transcript = os.path.join(base_convo_output_dir, "agent_2_full_transcript.json")

    agent_1_full_transcript_pdf = os.path.join(base_convo_output_dir, "agent_1_full_transcript.pdf")
    agent_2_full_transcript_pdf = os.path.join(base_convo_output_dir, "agent_2_full_transcript.pdf")

    convo_transcript = os.path.join(base_convo_output_dir, "convo.json")
    convo_transcript_pdf = os.path.join(base_convo_output_dir, "convo.pdf")
    convo_transcript_summ = os.path.join(transcript_dir, "0_base_convo.json")
    convo_transcript_summ_pdf = os.path.join(transcript_dir, "0_base_convo.pdf")

    audio_outputs_dir = os.path.join(base_convo_output_dir,"audio_out")
    os.makedirs(audio_outputs_dir,exist_ok=True)

    # Check if already processed  - if so load and return 
    if all(os.path.exists(f) for f in [agent_1_full_transcript_pdf, agent_2_full_transcript_pdf, agent_1_full_transcript, agent_2_full_transcript, convo_transcript, convo_transcript_pdf, convo_transcript_summ, convo_transcript_summ_pdf]):
        with open(agent_1_full_transcript, 'r') as f:
            agent_1_transcript = json.load(f)
        with open(agent_2_full_transcript, 'r') as f:
            agent_2_transcript = json.load(f)
        with open(convo_transcript, 'r') as f:
            convo = json.load(f)
        print("Base convo already processed - loading saved files. ")
        return {"agent_1_full_transcript": agent_1_transcript, 
                "agent_2_full_transcript": agent_2_transcript, 
                "convo_transcript": convo, 
                "convo_transcript_fpath": convo_transcript_summ}

    # Othewise produce the files 


    # Get all dynamic prompts 
    agent_1_full_description = get_agents_full_description(agent1, current_year) 
    agent_1_full_descriptio_counterpart = get_agents_counterpart_full_description(agent1, current_year) 
    agent_2_full_description = get_agents_full_description(agent2, current_year) 
    agent_2_full_descriptio_counterpart = get_agents_counterpart_full_description(agent2, current_year) 
    inductive_conversation = get_final_conversation(agent1, agent2, date, current_year)
    agent1_syntax, agent2_syntax = get_inductive_syntax_base_conversation(agent1, agent2)


    # Setup percentage tracking 
    percent_complete = 0
    turn_count = 0
    max_turns = world_orchestrator.base_conversation_steps

    # Obj for convo 
    convo_transcript_list = []

    # Iterate through 
    response_count = 0 
    while turn_count < max_turns+1:
        if turn_count == 0: 
            # Setup initial conversation dictionaries 
            agent_1_conversation = [{"role":"system","content":f"{EXPERIMENTAL_DESCRIPTION}\n Behavior Expectations {BEHAVIOR_EXPECTATIONS}\n Your role: {agent_1_full_description}\n Description of your counterpart that you have fromed {agent_1_full_descriptio_counterpart} Details about this conversation: {inductive_conversation}\n Specific Instructions about format: {agent1_syntax}\n"}]
            agent_2_conversation = [{"role":"system","content":f"{EXPERIMENTAL_DESCRIPTION}\n Behavior Expectations {BEHAVIOR_EXPECTATIONS}\n Your role: {agent_2_full_description}\n Description of your counterpart that you have fromed {agent_2_full_descriptio_counterpart} Details about this conversation: {inductive_conversation}\n Specific Instructions about format: {agent2_syntax}\n"}]
            agent_2_conversation.append({"role":"user","content":f"[PERCENT:{percent_complete}%] [START]"})


        # Process agent_2's conversation  -- this should return agent2 direct response 
        # Keep only the first message and last two for agent_2_conversation
        if len(agent_2_conversation) > 3:
            agent_2_conversation = [agent_2_conversation[0]] + agent_2_conversation[-1:]

        # Keep only the first message and last two for agent_1_conversation
        if len(agent_1_conversation) > 3:
            agent_1_conversation = [agent_1_conversation[0]] + agent_1_conversation[-1:]
        print(agent_2_conversation)
        # Pause for 1 minute
        # time.sleep(60)
        agent_2_response = llm_obj.call_audio(agent_2_conversation,audio_outputs_dir,response_count,voice_name="onyx")
        response_count += 1
        # Randomly simulate interruption
        if random.random() < random_cut_off and percent_complete < 90:
            # Randomly choose a character number between 10 and 70
            cut_off_length = random.randint(55, 95)
            agent_2_response = agent_2_response[:cut_off_length] + " [INTERRUPTION]"

        convo_transcript_list.append({f"{agent2.config.name}":f"{agent_2_response.replace('[RESPONSE]', '')}"})
        agent_1_conversation.append({"role":"user","content":f"[PERCENT:{percent_complete}%] {agent_2_response}"})
        agent_2_conversation.append({"role":"assistant","content":agent_2_response})

        # Now do reasoning step - agent_1 gets to think about how to respond - so we update conversation 
        agent_1_conversation = rag_reasoning_light(agent1, agent_1_conversation, llm_obj)
        # Get agent1 response 
        agent_1_response = llm_obj.call_audio(agent_1_conversation,audio_outputs_dir,response_count)
        response_count += 1
        # Randomly simulate interruption
        if random.random() < random_cut_off and percent_complete < 90: 
            # Randomly choose a character number between 10 and 70
            cut_off_length = random.randint(55, 95)
            agent_1_response = agent_1_response[:cut_off_length] + " [INTERRUPTION]"

        convo_transcript_list.append({f"{agent1.config.name}":f"{agent_1_response.replace('[RESPONSE]', '')}"})
        # Append to both agents convos 
        agent_2_conversation.append({"role":"user","content":f"[PERCENT:{percent_complete}%] {agent_1_response}"})
        agent_1_conversation.append({"role":"assistant","content":agent_1_response})

        # Now do reasoning step - agent_1 gets to think about how to respond - so we update conversation 
        agent_2_conversation = rag_reasoning_light(agent2, agent_2_conversation, llm_obj)


        # Update turn count and percent complete
        turn_count += 1
        percent_complete = int((turn_count / max_turns) * 100)
        print(turn_count)



    # Generate PDF and json of each agents convo, and the just transcript (no system prompt - just the actual back and forth of the conversation)
    utils.create_conversation_pdf_from_messages(agent_1_conversation, f"AGENT1 - Full Transcript  on {date}, Year {current_year}", agent_1_full_transcript_pdf)
    utils.create_conversation_pdf_from_messages(agent_2_conversation, f"AGENT2 - Full Transcript on {date}, Year {current_year}", agent_2_full_transcript_pdf)

    utils.create_conversation_pdf(convo_transcript_list, f"Conversation on {date}, Year {current_year}", convo_transcript_pdf)
    utils.create_conversation_pdf(convo_transcript_list, f"Conversation on {date}, Year {current_year}", convo_transcript_summ_pdf)
    with open(convo_transcript, "w") as f:
        json.dump(convo_transcript_list, f, indent=2)

    with open(convo_transcript_summ, "w") as f:
        json.dump(convo_transcript_list, f, indent=2)

    with open(agent_1_full_transcript, "w") as f:
        json.dump(agent_1_conversation, f, indent=2)

    with open(agent_2_full_transcript, "w") as f:
        json.dump(agent_2_conversation, f, indent=2)


    # Stitch together audio files
    # Get all wav files in the audio_outputs_dir
    wav_files = [f for f in os.listdir(audio_outputs_dir) if f.endswith('.wav')]
    # Sort the files based on their integer filenames
    sorted_wav_files = sorted(wav_files, key=lambda x: int(x.split('.')[0]))
    # Initialize an empty AudioSegment
    combined = AudioSegment.empty()
    # Iterate through sorted files and append to the combined AudioSegment
    for wav_file in sorted_wav_files:
        audio_path = os.path.join(audio_outputs_dir, wav_file)
        segment = AudioSegment.from_wav(audio_path)
        combined += segment

    # Export the combined audio
    output_path = os.path.join(audio_outputs_dir, "combined_conversation.wav")
    combined.export(output_path, format="wav")

    print(f"Combined audio saved to: {output_path}")



    return {"agent_1_full_transcript": agent_1_conversation, 
            "agent_2_full_transcript": agent_2_conversation, 
            "convo_transcript": convo_transcript_list,
            "convo_transcript_fpath": convo_transcript_summ}




#