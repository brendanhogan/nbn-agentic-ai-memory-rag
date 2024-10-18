"""
This project simulates long-term conversations between two AI agents, Willard and Jimmy, over a 50-year period.

Key features:
1. Agents converse 4 times a year, building a rich history of interactions.
2. After each conversation, agents reflect and store facts about each other.
3. Agents update their descriptions of each other based on these interactions.
4. A hierarchical reflection system is used to guide future conversations.
5. An orchestrator acts as a world model, introducing life events and challenges.

The system uses a Retrieval Augmented Generation (RAG) approach to maintain context and evolve the agents' relationship over time. This allows for more dynamic and context-aware conversations as the simulation progresses.

A few things to note about the code:
- I tried to make it flexible, but some stuff is still hard-coded. It's hard to make everything generic!
- The conversation schedule, memory system, and number of agents are set in stone for now.
- The prompts and memory parts are pretty tied to how i've set things up. 

I tried to make these choices to keep things working well, easy to understand, and not take forever to build.
"""
import os
import argparse

import llm
import agents
import audiogen
import embedding
import agentconfigs
import orchestrator
import conversations

# Set up argument parser
parser = argparse.ArgumentParser(description="Configure conversation simulation parameters")
parser.add_argument("--output_dir", type=str, default="output", help="Output directory for generated files")
parser.add_argument("--agent_1_config_name", type=str, default="willard", help="Configuration name for the first agent")
parser.add_argument("--agent_2_config_name", type=str, default="jimmy", help="Configuration name for the second agent")
parser.add_argument("--number_of_years", type=int, default=1, help="Number of years to simulate conversations")
parser.add_argument("--audioclass_name", type=str, default="OpenAIAudioGen", help="Name of the audio generation class")
parser.add_argument("--orchestratorclass_name", type=str, default="FriendConvoOrchestrator", help="Name of the orchestrator class")
parser.add_argument("--llm_name", type=str, default="gpt4o", help="Name of the language model to use")
parser.add_argument("--embedding_name", type=str, default="OpenAIEmbedding", help="Name of the embedding model to use")
parser.add_argument("--generate_audio", action="store_true", help="Flag to generate audio (default: False)")

# Parse arguments
args = parser.parse_args()


# Setup output dirs, and make sure exists 
all_transcript_dir = os.path.join(args.output_dir, "transcripts")  # Holds pdfs, and json of just conversations
all_audio_dir = os.path.join(args.output_dir, "audio")  # Holds all final audio files for each conversation, and reflection

# Create directories if they don't exist
os.makedirs(all_transcript_dir, exist_ok=True)
os.makedirs(all_audio_dir, exist_ok=True)

# Setup llm 
llm_obj = llm.get_llm(args.llm_name)

# Setup embeddings 
embed_obj = embedding.get_embedding_obj(args.embedding_name)

# Setup agents (within agents RAG is dealt with) 
agent_1_config = agentconfigs.get_agent_config(args.agent_1_config_name)
agent_2_config = agentconfigs.get_agent_config(args.agent_2_config_name)
agent1 = agents.Agent(args.output_dir, agent_1_config, embed_obj, llm_obj, agent_2_config)
agent2 = agents.Agent(args.output_dir, agent_2_config, embed_obj, llm_obj, agent_1_config)


# Setup audio class 
audio_generator = audiogen.get_audiogen(args.audioclass_name)
audio_generator.add_agent(agent1)
audio_generator.add_agent(agent2)
audio_generator.assign_voices()

# Setup orchestrator 
world_orchestrator = orchestrator.get_orchestrator(args.orchestratorclass_name, llm_obj, [agent1,agent2])

# Now iterate through and have conversations 
for year_idx in range(args.number_of_years): 
    current_year = year_idx + 2024
    for k, date in enumerate(['January 1st', 'April 1st', 'July 1st', 'October 1st']):
        date_int = year_idx*10+k
        if year_idx == 0 and date == "January 1st": 
            # Then just base case, have no memories to reflect on 
            base_convo_output_folder = os.path.join(args.output_dir, "conversations", f"{date_int}_base_convo")
            os.makedirs(base_convo_output_folder, exist_ok=True)
            convo_dict = conversations.base_conversation(all_transcript_dir, base_convo_output_folder, world_orchestrator, llm_obj, agent1, agent2, current_year, date)
            print(f"{date} {current_year} (Base) Conversation finished")
        else: 
            # If we are inductive case - first we need to use the orchestrators to develop a high level story line 
            orchestrator_output_dir = os.path.join(args.output_dir,"orchestrator_outs",f"{date_int}")
            os.makedirs(orchestrator_output_dir, exist_ok=True)
            story_line, event = world_orchestrator.generate_storyline(orchestrator_output_dir)
            # Inductive conversation (incorporates memories)
            base_convo_output_folder = os.path.join(args.output_dir, "conversations", f"{date_int}")
            os.makedirs(base_convo_output_folder, exist_ok=True)
            convo_dict = conversations.inductive_conversation(all_transcript_dir, base_convo_output_folder, world_orchestrator,llm_obj, story_line, agent1, agent2, current_year, date, event)
            print(f"{date} {current_year} Conversation finished")

        # Reflect on convo, will update rags of both agents 
        # Update with I/partner facts, reflections and deep reflections  - only from what was actually said 
        # Updates interal memory (RAG) system  - provide int system for date 
        agent_1_reflection_file = agent1.reflect(convo_dict['convo_transcript'], current_year, date, date_int) # This saves reflection to agent path 
        agent_2_reflection_file = agent2.reflect(convo_dict['convo_transcript'], current_year, date, date_int) 

        # Change to audio - every iteration will be three files to make into audo 
        # 1. the actual conversation 
        # 2. agent 1's reflection 
        # 3. agent 2's reflection 
        if args.generate_audio:
            print("Processing audio files")
            audio_generator.transcribe_and_save(convo_dict['convo_transcript_fpath'], os.path.join(args.output_dir,"audio",f"{date_int}_convo.mp3")) 
            audio_generator.transcribe_and_save(agent_1_reflection_file, os.path.join(args.output_dir,"audio",f"{date_int}_agent1reflect.mp3")) 
            audio_generator.transcribe_and_save(agent_2_reflection_file, os.path.join(args.output_dir,"audio",f"{date_int}_agent2reflect.mp3")) 

            

# Now have final life reflection call
print("Processing final conversation")
base_convo_output_folder = os.path.join(args.output_dir, "conversations", f"final")
os.makedirs(base_convo_output_folder, exist_ok=True)
convo_dict = conversations.final_conversation(all_transcript_dir, base_convo_output_folder, world_orchestrator,llm_obj, agent1, agent2, current_year, date)
print("fin")





#