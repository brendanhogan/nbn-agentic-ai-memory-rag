"""
This module contains the Agent class, which represents an AI agent in a simulated conversation experiment.
The Agent class manages the agent's knowledge, reflections, and interactions with its counterpart.
It utilizes Retrieval-Augmented Generation (RAG) systems to store and retrieve information,
and implements a reflection process after each conversation to update the agent's understanding
of itself and its counterpart.
"""
from typing import List, Tuple, Dict, Any, Optional
import os
import json


import utils
from rag import LMRRAG, UtilityRAG
from embedding import Embedding

class Agent:
    """
    Represents an AI agent with a rich simulated backstory and memory system.

    This class manages the agent's knowledge, reflections, and interactions with its counterpart.
    It utilizes six RAG systems (three for itself and three for its counterpart) to store facts,
    reflections, and deep reflections. After each conversation, the agent generates new insights,
    updates its RAG systems, and refines its understanding of itself and its counterpart.

    Attributes:
        output_dir (str): Directory for output files.
        config (Any): Configuration object for the agent.
        embedding_model (Embedding): Model for generating embeddings.
        llm_obj (Any): Language model object for generating responses.
        self_rag (LMRRAG): RAG system for the agent's own information.
        counterpart_rag (LMRRAG): RAG system for the counterpart's information.
        description_of_self (str): Current description of the agent.
        description_of_counterpart (str): Current description of the counterpart.
        counterpart_name (str): Name of the counterpart agent.
    """

    def __init__(self, output_dir: str, agent_config_obj: Any, embedding_model: Embedding, llm_obj: Any, counterpart_config: Any):
        self.output_dir = output_dir
        self.config = agent_config_obj
        self.embedding_model = embedding_model
        self.llm_obj = llm_obj

        # Initialize RAGs
        self.self_rag = LMRRAG(UtilityRAG, self.embedding_model, f"{output_dir}/{self.config.name}_self_rag")
        self.counterpart_rag = LMRRAG(UtilityRAG, self.embedding_model, f"{output_dir}/{self.config.name}_counterpart_rag")

        # Initialize descriptions
        self.description_of_self = self.config.description
        self.description_of_counterpart = counterpart_config.description 
        self.counterpart_name = counterpart_config.name 

    def reflect(self, transcript: str, current_year: int, date: str, date_int: int) -> str:
        """
        Process a conversation transcript and update the agent's knowledge and reflections.

        This method generates facts and reflections about the agent and its counterpart,
        updates the RAG systems, generates and answers deep reflection questions,
        and updates the agent's descriptions of itself and its counterpart.

        Args:
            transcript (str): The conversation transcript to reflect on.
            current_year (int): The current year in the simulation.
            date (str): The date of the conversation.
            date_int (int): Integer representation of the date.

        Returns:
            str: Path to the JSON file containing the consciousness reflection transcript.
        """
        # Setup reflection outputdir
        transcript_dir = os.path.join(self.output_dir, "transcripts")
        ref_output_dir = os.path.join(self.output_dir, "reflections", f"{date_int}_reflection",f"{self.config.name}_refs") 
        rag_output_dir_self = os.path.join(self.output_dir, "reflections", f"{date_int}_reflection",f"{self.config.name}_self_rags") 
        rag_output_dir_counter = os.path.join(self.output_dir, "reflections", f"{date_int}_reflection",f"{self.config.name}_counter_rags") 
        transcript_dir_json_path = os.path.join(transcript_dir, f"{date_int}_{self.config.name}_consciousness_reflection_transcript.json")
        os.makedirs(ref_output_dir, exist_ok=True)
        os.makedirs(rag_output_dir_self, exist_ok=True)
        os.makedirs(rag_output_dir_counter, exist_ok=True)

        # Try to load RAG models from file - if succesful then done, otherwise have to run it all 
        self_loaded = self.self_rag.load_from_file(rag_output_dir_self)
        counter_loaded = self.counterpart_rag.load_from_file(rag_output_dir_counter)
        if self_loaded and counter_loaded: 
            # Load updated descriptions
            with open(os.path.join(ref_output_dir, f"{self.config.name}_updated_description.txt"), "r") as f:
                self.config.description = f.read().strip()
            with open(os.path.join(ref_output_dir, f"{self.counterpart_name}_updated_description.txt"), "r") as f:
                self.counterpart_description = f.read().strip()

            # Update config description
            self.config.description = self.description_of_self
            print(f"Updates/ RAGS for {date} {current_year} already done, files loaded and now skipping ")
            return transcript_dir_json_path


        # Generate facts and reflections
        self_facts, self_reflections, self_fact_ref_convo = self._generate_facts_and_reflections(transcript, is_self=True)
        counterpart_facts, counterpart_reflections, counterpart_fact_ref_convo = self._generate_facts_and_reflections(transcript, is_self=False)

        # Save to pdf 
        utils.create_conversation_pdf_from_messages(self_fact_ref_convo,f"{self.config.name}s_fact_refs_about_self",os.path.join(ref_output_dir,f"{self.config.name}s_fact_refs_about_self.pdf"))
        utils.create_conversation_pdf_from_messages(counterpart_fact_ref_convo,f"{self.config.name}s_fact_refs_about_{self.counterpart_name}",os.path.join(ref_output_dir,f"{self.config.name}s_fact_refs_about_{self.counterpart_name}.pdf"))

        # # Add to RAGs
        self.self_rag.add_facts(self_facts, [date_int] * len(self_facts))
        self.self_rag.add_reflections(self_reflections, [date_int] * len(self_reflections))
        self.counterpart_rag.add_facts(counterpart_facts, [date_int] * len(counterpart_facts))
        self.counterpart_rag.add_reflections(counterpart_reflections, [date_int] * len(counterpart_reflections))

        # # Generate and process deep reflection questions
        questions, generated_questions_convo, answer_convos, deep_reflections = self._process_deep_reflections(transcript, date_int, is_self=True)
        cp_questions, cp_generated_questions_convo, cp_answer_convos, cp_deep_reflections = self._process_deep_reflections(transcript, date_int, is_self=False)
        # Log questions and deep reflections
        with open(os.path.join(ref_output_dir, f"{self.config.name}_self_questions.txt"), "w") as f:
            f.write("\n".join(questions))
        with open(os.path.join(ref_output_dir, f"{self.config.name}_counterpart_questions.txt"), "w") as f:
            f.write("\n".join(cp_questions))
        with open(os.path.join(ref_output_dir, f"{self.config.name}_self_deep_reflections.txt"), "w") as f:
            f.write("\n".join(deep_reflections))
        with open(os.path.join(ref_output_dir, f"{self.config.name}_counterpart_deep_reflections.txt"), "w") as f:
            f.write("\n".join(cp_deep_reflections))

        # Save conversation PDFs
        utils.create_conversation_pdf_from_messages(generated_questions_convo, f"{self.config.name}_self_questions_generation", os.path.join(ref_output_dir, f"{self.config.name}_self_questions_generation.pdf"))
        utils.create_conversation_pdf_from_messages(cp_generated_questions_convo, f"{self.config.name}_counterpart_questions_generation", os.path.join(ref_output_dir, f"{self.config.name}_counterpart_questions_generation.pdf"))
        
        for i, convo in enumerate(answer_convos):
            utils.create_conversation_pdf_from_messages(convo, f"{self.config.name}_self_deep_reflection_{i+1}", os.path.join(ref_output_dir, f"{self.config.name}_self_deep_reflection_{i+1}.pdf"))
        for i, convo in enumerate(cp_answer_convos):
            utils.create_conversation_pdf_from_messages(convo, f"{self.config.name}_counterpart_deep_reflection_{i+1}", os.path.join(ref_output_dir, f"{self.config.name}_counterpart_deep_reflection_{i+1}.pdf"))

        # Create summary file
        summary = f"""Facts {self.config.name} learned about self:
            {json.dumps(self_facts, indent=2)}

            Facts {self.config.name} learned about {self.counterpart_name}:
            {json.dumps(counterpart_facts, indent=2)}

            Reflections about {self.config.name}:
            {json.dumps(self_reflections, indent=2)}

            Reflections about {self.counterpart_name}:
            {json.dumps(counterpart_reflections, indent=2)}

            Deep reflection questions and answers for {self.config.name}:
            {json.dumps(list(zip(questions, deep_reflections)), indent=2)}

            Deep reflection questions and answers for {self.counterpart_name}:
            {json.dumps(list(zip(cp_questions, cp_deep_reflections)), indent=2)}
            """

        with open(os.path.join(ref_output_dir, f"{self.config.name}_reflection_summary.txt"), "w") as f:
            f.write(summary)
        # # Update descriptions
        updated_description_self, description_self_convo = self._update_descriptions(summary, is_self=True)
        updated_description_counterpart, description_counterpart_convo = self._update_descriptions(summary, is_self=False)

        # Save updated descriptions as text files
        with open(os.path.join(ref_output_dir, f"{self.config.name}_updated_description.txt"), "w") as f:
            f.write(updated_description_self)
        with open(os.path.join(ref_output_dir, f"{self.counterpart_name}_updated_description.txt"), "w") as f:
            f.write(updated_description_counterpart)

        # Save description update conversations as PDFs
        utils.create_conversation_pdf_from_messages(description_self_convo, f"{self.config.name}_description_update", os.path.join(ref_output_dir, f"{self.config.name}_description_update.pdf"))
        utils.create_conversation_pdf_from_messages(description_counterpart_convo, f"{self.counterpart_name}_description_update", os.path.join(ref_output_dir, f"{self.counterpart_name}_description_update.pdf"))

        # Last thing write out, and save all rags 
        self.self_rag.write_and_save(rag_output_dir_self)
        self.counterpart_rag.write_and_save(rag_output_dir_counter)

        # Now make conversation with 'consciousnes' for logging/audio generation purposes 
        convos, transcript_convo = self._generate_consciousness_reflection(summary, current_year)
        # Save convos and transcript_convo as JSON and PDF in ref_output_dir
        convos_json_path = os.path.join(ref_output_dir, f"{self.config.name}_consciousness_reflection_convos.json")
        transcript_convo_json_path = os.path.join(ref_output_dir, f"{self.config.name}_consciousness_reflection_transcript.json")
        
        with open(convos_json_path, 'w') as f:
            json.dump(convos, f, indent=2)
        
        with open(transcript_convo_json_path, 'w') as f:
            json.dump(transcript_convo, f, indent=2)
        
        utils.create_conversation_pdf_from_messages(convos, f"{self.config.name}_consciousness_reflection_convos", os.path.join(ref_output_dir, f"{self.config.name}_consciousness_reflection_convos.pdf"))
        utils.create_conversation_pdf(transcript_convo, f"{self.config.name}_consciousness_reflection_transcript", os.path.join(ref_output_dir, f"{self.config.name}_consciousness_reflection_transcript.pdf"))
        
        # Save transcript_convo as JSON and PDF in transcript_dir
        
        with open(transcript_dir_json_path, 'w') as f:
            json.dump(transcript_convo, f, indent=2)
        
        utils.create_conversation_pdf(transcript_convo, f"{self.config.name}_consciousness_reflection_transcript", os.path.join(transcript_dir, f"{date_int}_{self.config.name}_consciousness_reflection_transcript.pdf"))

        return transcript_dir_json_path

    def _generate_consciousness_reflection(self, summary: str, current_year: int) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]: 

        system_prompt = f"""
        You are part of an artificial intelligence experiment to see if, by building two AI agents with rich simulated human backstories 
        and a memory system, and simulating 50 years of phone conversations between them, any interesting emergent behavior will occur. 
        We aim to explore whether it will feel like the AI agents are truly learning, deepening their own understanding of themselves, and 
        building real, deep personalities.


        At this stage, you have just finished a conversation. You are playing the role of {self.config.name}. 
        You then looked at the conversation, and reflected on it, and came up with conclusions about yourself, 
        as well as your counterpart {self.counterpart_name} - these include facts reflection and deep reflections 
        about you and your counterpart. 

        In this final step you will be given the summary of your findings frm the reflection and be prompted by your 
        'consciousness' to discuss and summarize your findings. 

        In your response you should stay purely in character as {self.config.name} and give a nice response summarizing 
        what you learned about your counterpart and friend. 

        The year is {current_year} and you started talking in year 2024, so you can reflect on how young/old you are 
        and how much you have been talking. 

        """
        prompt_0 = f"""
        Here is the summary of your ({self.config.name}) reflections about yourself and your friend {self.counterpart_name}.
        {summary}

        In your response, first spend about 25% of your answer reflecting on what you learned about {self.counterpart_name} - both factual information and deeper insights into their personality based on the reflections.

        Then, for the remaining 75% of your answer, focus on self-reflection. What did you learn about yourself in this conversation? Express yourself colloquially, using informal language, slang, and filler words like "um" and "hmm". Talk like a regular person would in casual conversation, not in a formal or AI-like manner. Really let your personality shine through as you ponder what this interaction revealed about you.
       
        The next message on will restate the setting a little, and be frames as your consciousness. 
        Answer fully in character, like you are the character talking to themselves in their own mind. 
        """
        prompt_1 = f"""
        {self.config.name}, this is your consciousness. Let's reflect on your recent conversation with {self.counterpart_name} and the insights you've gained.

        Now, take a moment to ponder. How has this conversation shaped your understanding of yourself and {self.counterpart_name}? Consider how your relationship has evolved since those first exchanges back in 2024. It's now {current_year} - think about the passage of time, the growth you've both experienced, the shared memories you've created.

        What stands out to you most? How do you feel your perspectives have shifted? Are there any new realizations about your life, your goals, or your connection with {self.counterpart_name}? 

        Let your thoughts flow freely. This is a space for deep introspection and honest reflection.
        """


        convos = [{"role":"system","content":system_prompt}]
        convos.append({"role":"user","content":prompt_0})
        convos.append({"role":"user","content":prompt_1})
        llm_response = self.llm_obj.call(convos)
        convos.append({"role":"assistant","content":llm_response})


        return_convo = []
        return_convo.append({"consciousness":prompt_1})
        return_convo.append({f"{self.config.name}":llm_response})

        return convos, return_convo

    def _generate_facts_and_reflections(self, transcript: str, is_self: bool) -> Tuple[List[str], List[str], List[Dict[str, str]]]:
        # Prepare the prompt for the LLM
        subject_name = self.config.name if is_self else self.counterpart_name
        system_prompt = f"""
        You are an expert analyzer and reflector of conversations, skilled at extracting key insights and observations.
        You are taking on the role of {self.config.name} for an AI experiement about ability of emergent behavior to happen with AI agents. 
        You are doing a post-analysis of your most recent conversation with {self.counterpart_name} in order to form memories for a 
        retrieval augmented generation memory system. 

        """
        prompt = f"""

        Based on the following conversation transcript, generate 3 factual statements and 3 reflective statements about {subject_name}.
        Facts should be objective observations, while reflections should be more interpretive or emotional insights.

        Transcript:
        {transcript}

        Output your analysis in JSON format as follows:
        {{
            "facts": [
                "Fact 1",
                "Fact 2",
                "Fact 3"
            ],
            "reflections": [
                "Reflection 1",
                "Reflection 2",
                "Reflection 3"
            ]
        }}
        It is extremely important you only answer with json - as this will be parsed by python. 
        """

        # Use the LLM to generate the response
        convos = [{"role":"system","content":system_prompt}]
        convos.append({"role":"user","content":prompt})
        llm_response = self.llm_obj.call(convos)

        # Remove anything before the first '{' and after the last '}'
        llm_response = llm_response[llm_response.find('{'):llm_response.rfind('}')+1]
        convos.append({"role":"assistant","content":llm_response})
        parsed_response = json.loads(llm_response)
        
        facts = parsed_response['facts']
        reflections = parsed_response['reflections']

        return facts, reflections, convos

    def _process_deep_reflections(self, transcript: str, date_int: int, is_self: bool) -> Tuple[List[str], List[Dict[str, str]], List[List[Dict[str, str]]], List[str]]:
        # Generate deep reflection questions
        questions, generated_questions_convo = self._generate_deep_reflection_questions(transcript, is_self)

        rag = self.self_rag if is_self else self.counterpart_rag

        answer_convos = []
        deep_reflections_new = []
        for question in questions:
            # Retrieve relevant reflections and deep reflections
            reflections = rag.get_reflections(question)
            deep_reflections = rag.get_deep_reflections(question)

            # Generate answer using an LLM
            answer, tmp_convo = self._generate_deep_reflection_answer(question, reflections, deep_reflections, is_self)
            answer_convos.append(tmp_convo)
            deep_reflections_new.append(answer)
            # Save the answer as a new deep reflection
            rag.add_deep_reflections([answer], [date_int])

        
        return questions, generated_questions_convo, answer_convos, deep_reflections_new

    def _generate_deep_reflection_questions(self, transcript: str, is_self: bool) -> Tuple[List[str], List[Dict[str, str]]]:
        subject_name = self.config.name if is_self else self.counterpart_name
        system_prompt = f"""
        You are an expert psychologist and conversation analyst, skilled at generating deep, thought-provoking questions.
        You are taking on the role of {self.config.name} for an AI experiment about the ability of emergent behavior to happen with AI agents.
        You are analyzing your most recent conversation with {self.counterpart_name} to generate deep reflection questions.
        """
        prompt = f"""
        Based on the following conversation transcript, generate 3 deep, thought-provoking questions about {subject_name}.
        These questions should encourage introspection and explore complex aspects of personality, relationships, or personal growth.

        {"If you're asking about " + self.counterpart_name + ", always mention their name in the questions (e.g., 'Why does " + self.counterpart_name + " feel this way?')." if not is_self else ""}

        Transcript:
        {transcript}

        Output your questions in JSON format as follows:
        {{
            "questions": [
                "Question 1",
                "Question 2",
                "Question 3"
            ]
        }}
        It is extremely important you only answer with json - as this will be parsed by python.
        """

        convos = [{"role": "system", "content": system_prompt}]
        convos.append({"role": "user", "content": prompt})
        llm_response = self.llm_obj.call(convos)

        llm_response = llm_response[llm_response.find('{'):llm_response.rfind('}')+1]
        convos.append({"role": "assistant", "content": llm_response})
        parsed_response = json.loads(llm_response)
        
        return parsed_response['questions'], convos

    def _generate_deep_reflection_answer(self, question: str, reflections: List[str], deep_reflections: List[str], is_self: bool) -> Tuple[str, List[Dict[str, str]]]:
        subject_name = self.config.name if is_self else self.counterpart_name
        system_prompt = f"""
        You are an expert in self-reflection and personal growth, skilled at synthesizing insights from various sources.
        You are taking on the role of {self.config.name} for an AI experiment about the ability of emergent behavior to happen with AI agents.
        You are answering a deep reflection question about {subject_name} based on previous reflections and deep reflections.
        """
        prompt = f"""
        Consider the following deep reflection question about {subject_name}:
        {question}

        Use the following previous reflections and deep reflections to inform your answer (may be blank if not enough memories yet):
        Reflections: {reflections}
        Deep Reflections: {deep_reflections}

        Provide a thoughtful, introspective answer to the question. Your response should be a single paragraph of 3-5 sentences.

        {"If you're answering about " + self.counterpart_name + ", always mention their name in the answer, and answer as if its your best guess about them, not as if you are answering for them." if not is_self else ""}
        """

        convos = [{"role": "system", "content": system_prompt}]
        convos.append({"role": "user", "content": prompt})
        llm_response = self.llm_obj.call(convos)
        convos.append({"role": "assistant", "content": llm_response})

        return llm_response.strip(), convos
 
    def _update_descriptions(self, summary: str, is_self: bool) -> Tuple[str, List[Dict[str, str]]]:
        subject = "self" if is_self else "counterpart"
        current_description = self.config.description if is_self else self.description_of_counterpart
        
        
        system_prompt = f"""
        You are an AI assistant helping to update the description of an agent or their counterpart.
        You should be extremely conservative in making changes, only adding or modifying information
        if there is a direct conflict or if the new information is very critical.
        The description should remain about the same length.
        """
        
        user_prompt = f"""
        Current {subject} description:
        {current_description}

        New information from summary:
        {summary}

        Please update the {subject} description based on this new information.
        Only make changes if absolutely necessary and keep the length similar.

        If no changes are necessary return exactly the same text. 

        Whatever text you return will be exactly used to update the description, so only return the updated 
        information, nothing else. 

        """

        convos = [{"role": "system", "content": system_prompt}]
        convos.append({"role": "user", "content": user_prompt})
        updated_description = self.llm_obj.call(convos).strip()
        convos.append({"role": "assistant", "content": updated_description})

        if is_self:
            self.config.description = updated_description
        else:
            self.counterpart_description = updated_description

        return updated_description, convos
