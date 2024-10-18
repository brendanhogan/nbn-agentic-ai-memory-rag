# NBN - 50 Years of AI Conversations: Agentic AI with RAG for Memories and Reflection

## Table of Contents
- [Project Description](#project-description)
- [Code Structure](#code-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Command-line Arguments](#command-line-arguments)
- [Module Descriptions](#module-descriptions)
  - [main.py](#mainpy)
  - [agentconfigs.py](#agentconfigspy)
  - [agents.py](#agentspy)
  - [orchestrator.py](#orchestratorpy)
  - [rag.py](#ragpy)
  - [embedding.py](#embeddingpy)
  - [conversations.py](#conversationspy)
- [Extending the Project](#extending-the-project)
- [Viewing Results](#viewing-results)
- [Notes](#notes)
- [Contributing](#contributing)
- [License](#license)


## Project Description

This project simulates long-term conversations between two AI agents over a 50-year period, demonstrating the potential for emergent behavior in AI systems. The 'conversations' are meant to be quartly phone calls, and can be transformed into real audio using OpenAI's new real time audio engine. It utilizes a Retrieval Augmented Generation (RAG) approach to maintain context and evolve the agents' relationship over time, allowing for more dynamic and context-aware conversations as the simulation progresses. The agents add to their respective RAGs, by reflecting after every conversation, asking what are the new facts they can learn, and higher level reflections, which then use the same RAG system to answer - building a heirarchy of reflections and personality features of the agents - very similar to [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442), which was a huge inspiration for this work. 

This project is part of a blog post exploring the implications of long-term AI interactions. For more details, please check out the full blog post:

[Read the full blog post](https://www.google.com)

Key features:
- Agents converse 4 times a year, building a rich history of interactions.
- After each conversation, agents reflect and store facts about each other.
- Agents update their descriptions of each other based on these interactions.
- A hierarchical reflection system guides future conversations.
- An orchestrator acts as a world model, introducing life events and challenges.

## Code Structure

The project consists of several key modules:

1. `main.py`: The entry point of the application, orchestrating the entire simulation process.
2. `agentconfigs.py`: Defines the initial configurations for the AI agents.
3. `agents.py`: Implements the Agent class, managing agent behavior, memory, and interactions.
4. `orchestrator.py`: Sets up the world model and dynamically adds events to the simulation.
5. `rag.py`: Handles the Retrieval Augmented Generation system for memory management.
6. `embedding.py`: Provides text embedding functionality for the RAG system.
7. `conversations.py`: Manages the conversation flow, including base case and inductive case scenarios.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/nbn-50years-ai-conversations.git
   cd nbn-50years-ai-conversations
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set the OPENAI_API_KEY environment variable:
   ```bash
   export OPENAI_API_KEY=your_api_key_here
   ```
   Note: Make sure to replace `your_api_key_here` with your actual OpenAI API key.

## Usage

To run the simulation, use the `main.py` script with appropriate arguments:
```bash
python main.py --output_dir output --agent_1_config_name willard --agent_2_config_name jimmy --number_of_years 50 --llm_name gpt4o --embedding_name OpenAIEmbedding
```

### Command-line Arguments

- `--output_dir`: Directory for generated files (default: "output")
- `--agent_1_config_name`: Configuration name for the first agent (default: "willard")
- `--agent_2_config_name`: Configuration name for the second agent (default: "jimmy")
- `--number_of_years`: Number of years to simulate conversations (default: 1)
- `--audioclass_name`: Name of the audio generation class (default: "OpenAIAudioGen")
- `--orchestratorclass_name`: Name of the orchestrator class (default: "FriendConvoOrchestrator")
- `--llm_name`: Name of the language model to use (default: "gpt4o")
- `--embedding_name`: Name of the embedding model to use (default: "OpenAIEmbedding")
- `--generate_audio`: Flag to generate audio (default: False)

## Module Descriptions

### main.py
Orchestrates the entire simulation, setting up agents, the orchestrator, and managing the conversation loop over the specified number of years.

### agentconfigs.py
Defines initial configurations for AI agents, including personality traits, background, and initial knowledge.

### agents.py
Implements the Agent class, which manages individual agent behavior, memory systems, and interaction capabilities. After every conversation, the agent forms facts, reflections, and deep reflections about itself and its counterpart. It updates its self-description and understanding of the other agent based on these insights. This continuous learning process allows the agents to evolve their personalities and relationships over time.

### orchestrator.py
Creates the world model, introducing dynamic events and challenges throughout the simulation to influence agent interactions. It randomly generates conversation scenarios, including types (good news, bad news, fight, or regular conversation), severity (moderate or severe), and specific reasons. This dynamic storyline generation ensures varied and engaging interactions between agents over the simulated years.

### rag.py
Implements a Retrieval Augmented Generation (RAG) system, managing the storage and retrieval of memories for each agent. The system maintains separate RAG instances for facts, reflections, and deep reflections. Each RAG utilizes embedding-based similarity search to retrieve the most relevant information, which is then sorted by recency to provide context-aware and up-to-date information for agent interactions.

### embedding.py
Provides text embedding functionality, crucial for the RAG system's ability to store and retrieve contextually relevant information. These embeddings enable efficient similarity-based searches across the agents' accumulated knowledge and experiences.

### conversations.py
Manages the flow of conversations, including both base case (initial) and inductive case (memory-informed) scenarios. In the inductive case, at every conversation step, it leverages the RAG system to pull in relevant information. This retrieval process considers both relevance (via embedding similarity) and recency (time-based sorting) to guide the agents' responses, ensuring that their dialogue is informed by past experiences and accumulated knowledge.

## Extending the Project

To extend the project:

1. Add new agent configurations in `agentconfigs.py`.
2. Implement new conversation types or modify existing ones in `conversations.py`.
3. Enhance the world model by adding new event types in `orchestrator.py`.
4. Improve the RAG system in `rag.py` for more sophisticated memory management.
5. Implement new embedding models in `embedding.py` for better text representation.

## Viewing Results

Results are stored in the specified output directory:

- Conversation transcripts: `output/transcripts/`
- Audio files (if generated): `output/audio/`
- Individual conversation outputs: `output/conversations/`
- Orchestrator outputs: `output/orchestrator_outs/`

To analyze the results:
1. Review the JSON and PDF files in the transcript directory for conversation content.
2. Examine the reflection files in each agent's directory to see how their understanding evolves.
3. Listen to the generated audio files (if enabled) for a more immersive experience.


## Notes on Project Structure

While developing this project, I aimed to create a modular and extensible system. The initial vision included the possibility of incorporating multiple agents in simultaneous conversations and defining various interaction frameworks. However, due to time constraints and the complexity involved, some aspects of the project remain more rigid than initially intended.

Key points to note:

1. Partial Modularity: Some components, like the embedding system and the RAG implementation, are designed with abstraction in mind, allowing for easier extension or replacement.

2. Hard-coded Elements: Certain aspects, such as the two-agent conversation structure and the specific reflection process, are more tightly integrated into the current implementation.

3. Conversation Flow: The conversation scheduling and progression are relatively fixed, making it challenging to introduce dynamic multi-agent scenarios without significant refactoring.

4. Agent Configuration: While new agent profiles can be added easily, the underlying agent behavior and interaction patterns are somewhat constrained by the current implementation.

5. Orchestrator Limitations: The world model and event generation system, while functional, have a predefined set of scenarios and don't easily accommodate complex, multi-agent dynamics.


## Contributing

Contributions to improve and extend this project are welcome. Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature.
3. Implement your changes.
4. Submit a pull request with a clear description of your improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.