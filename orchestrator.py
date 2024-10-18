"""
This module contains the orchestrator class responsible for managing inter-agent communication
and generating conversation storylines for AI agents. It provides functionality to:

1. Generate general and specialized conversation storylines
2. Retrieve and incorporate agent memories into the storylines
3. Save generated storylines and conversations to various file formats
4. Instantiate the appropriate orchestrator based on configuration

The module is designed to work with a pair of AI agents, simulating conversations over an
extended period, and incorporates a retrieval-augmented generation (RAG) system for enhanced
context and memory retrieval.
"""

import os
import json 
import random
from typing import Dict, List, Tuple, Any

import utils


class FriendConvoOrchestrator:
    """
    Orchestrates conversations between two AI agents, generating storylines and managing interactions.
    """

    def __init__(self, llm_obj: Any, agent1: Any, agent2: Any, number_of_years: int = 50) -> None:
        """
        Initialize the FriendConvoOrchestrator.

        Args:
            llm_obj (Any): The language model object used for generating responses.
            agent1 (Any): The first AI agent.
            agent2 (Any): The second AI agent.
            number_of_years (int): The number of years to simulate conversations for.
        """
        self.llm_obj = llm_obj
        self.agent1 = agent1
        self.agent2 = agent2 
        self.number_of_years = number_of_years

        self.base_conversation_steps = 8

    def generate_storyline(self, output_dir: str) -> Tuple[str, Dict[str, Any]]:
        """
        Generate and save both general and specialized storylines for a conversation.

        Args:
            output_dir (str): Directory to save the generated files.

        Returns:
            Tuple[str, Dict[str, Any]]: The specialized storyline and the general event dictionary.
        """
        # Generate general storyline
        event = self.general_conversation_storyline()
        storyline_file = os.path.join(output_dir, "generic_storyline.json")
        with open(storyline_file, "w") as f:
            json.dump(event, f, indent=2)

        # Generate specialized storyline
        convo, specialized_storyline = self.specialized_conversation_storyline(event)

        # Save conversation as JSON
        convo_json_path = os.path.join(output_dir, "specialized_conversation.json")
        with open(convo_json_path, "w") as f:
            json.dump(convo, f, indent=2)

        # Save conversation as PDF
        convo_pdf_path = os.path.join(output_dir, "specialized_conversation.pdf")
        utils.create_conversation_pdf_from_messages(convo, "Specialized Conversation", convo_pdf_path)

        # Save specialized storyline as text
        storyline_txt_path = os.path.join(output_dir, "specialized_storyline.txt")
        with open(storyline_txt_path, "w") as f:
            f.write(specialized_storyline)

        return specialized_storyline, event

    def general_conversation_storyline(self) -> Dict[str, Any]:
        """
        Generate a general conversation storyline by randomly selecting a conversation type and details.

        Returns:
            Dict[str, Any]: A dictionary containing the conversation type, severity, and reason.
        """
        conversation_types = {
            'good_news': 0.2,
            'bad_news': 0.2,
            'fight': 0.2,
            'regular_convo': 0.4
        }

        selected_type = random.choices(list(conversation_types.keys()), weights=list(conversation_types.values()))[0]

        severity = None
        reason = None

        if selected_type != 'regular_convo':
            severity = random.choices(['moderate', 'severe'], weights=[0.6, 0.4])[0]

        if selected_type == 'good_news':
            if severity == 'moderate':
                reasons = ['new job', 'new pet', 'small promotion', 'successful project']
            else:  # severe
                reasons = ['wedding', 'birth of child', 'major promotion', 'life-changing opportunity']
            reason = random.choice(reasons)

        elif selected_type == 'bad_news':
            if severity == 'moderate':
                reasons = ['minor health issue', 'job setback', 'financial difficulty', 'relationship problem']
            else:  # severe
                reasons = ['major health crisis', 'job loss', 'significant financial loss', 'death in family']
            reason = random.choice(reasons)

        elif selected_type == 'fight':
            if severity == 'moderate':
                reasons = ['disagreement over plans', 'misunderstanding', 'differing opinions']
            else:  # severe
                reasons = ['betrayal of trust', 'long-standing issue surfacing', 'fundamental value clash']
            reason = random.choice(reasons)

        return {
            'type': selected_type,
            'severity': severity,
            'reason': reason
        }

    def specialized_conversation_storyline(self, general_storyline: Dict[str, Any]) -> Tuple[List[Dict[str, str]], str]:
        """
        Generate a specialized conversation storyline based on the general storyline and agent memories.

        Args:
            general_storyline (Dict[str, Any]): The general storyline dictionary.

        Returns:
            Tuple[List[Dict[str, str]], str]: The conversation history and the specialized storyline.
        """
        # Synthesize the general storyline into a concise prompt for memory retrieval
        synthesis_prompt = f"""
        Synthesize the following conversation scenario into a single sentence or two. This will be used to retrieve relevant memories from an AI agent's memory system:

        Conversation type: {general_storyline['type']}
        Severity: {general_storyline['severity']}
        Reason: {general_storyline['reason']}
        """
        
        convo = [{"role": "system", "content": "You are an expert storyteller and scenario creator, specializing in crafting believable and engaging storylines for conversations between two individuals. Your ability to weave intricate narratives that feel authentic and relatable is unparalleled."}]
        convo.append({"role": "user", "content": synthesis_prompt})
        synthesis = self.llm_obj.call(convo)
        convo.append({"role": "assistant", "content": synthesis})

        # Retrieve relevant memories for both agents
        agent1_memories = self._retrieve_agent_memories(self.agent1, synthesis)
        agent2_memories = self._retrieve_agent_memories(self.agent2, synthesis)
        
        # Generate the specialized storyline
        storyline_prompt = f"""
        Based on the following general conversation scenario and retrieved memories from two AI agents, create a detailed 1-2 paragraph storyline for their upcoming conversation.

        General scenario:
        Type: {general_storyline['type']}
        Severity: {general_storyline['severity']}
        Reason: {general_storyline['reason']}

        Agent 1 ({self.agent1.config.name}) memories:
        {agent1_memories}

        Agent 2 ({self.agent2.config.name}) memories:
        {agent2_memories}

        If it's a regular conversation, provide a high-level overview of potential topics they might discuss.
        For good news or bad news, give specific details about the event or situation.
        If it's a fight, elaborate on the specific reasons for their disagreement.

        Your response should be a coherent 1-2 paragraph storyline that incorporates elements from both the general scenario and the agents' memories.
        
        Important: The conversation takes place over the phone. 
        """

        convo.append({"role": "user", "content": storyline_prompt})
        specialized_storyline = self.llm_obj.call(convo)
        convo.append({"role": "assistant", "content": specialized_storyline})

        return convo, specialized_storyline

    def _retrieve_agent_memories(self, agent: Any, query: str) -> str:
        """
        Retrieve relevant memories (facts, reflections, and deep reflections) for an agent.

        Args:
            agent (Any): The agent object to retrieve memories from.
            query (str): The query string to use for memory retrieval.

        Returns:
            str: A formatted string containing the retrieved memories.
        """
        facts_self = agent.self_rag.get_facts(query)
        reflections_self = agent.self_rag.get_reflections(query)
        deep_reflections_self = agent.self_rag.get_deep_reflections(query)
        
        facts_counterpart = agent.counterpart_rag.get_facts(query)
        reflections_counterpart = agent.counterpart_rag.get_reflections(query)
        deep_reflections_counterpart = agent.counterpart_rag.get_deep_reflections(query)
        
        memories = f"""
        Self Facts: {facts_self}
        Self Reflections: {reflections_self}
        Self Deep Reflections: {deep_reflections_self}
        Counterpart Facts: {facts_counterpart}
        Counterpart Reflections: {reflections_counterpart}
        Counterpart Deep Reflections: {deep_reflections_counterpart}
        """
        
        return memories

def get_orchestrator(orchestrator_class_name: str, llm_obj: Any, agents: List[Any], number_of_years: int = 50) -> FriendConvoOrchestrator:
    """
    Returns an initialized instance of the specified orchestrator class.

    Args:
        orchestrator_class_name (str): The name of the orchestrator class.
        llm_obj (Any): The language model object.
        agents (List[Any]): A list containing the two agent objects.
        number_of_years (int): The number of years for the conversation simulation.

    Returns:
        FriendConvoOrchestrator: An instance of the specified orchestrator class.

    Raises:
        NotImplementedError: If an unsupported orchestrator class name is provided.
    """
    if orchestrator_class_name == "FriendConvoOrchestrator":
        return FriendConvoOrchestrator(llm_obj, agents[0], agents[1], number_of_years)
    else:
        raise NotImplementedError(f"Orchestrator class '{orchestrator_class_name}' is not implemented.")