# LLM-Enhanced Wireless Optimization Framework for Secure Communications in LAWNs
<img width="1781" height="831" alt="image" src="https://github.com/user-attachments/assets/abce1dbb-eee4-497a-8fcc-80cfdac59758" />
\begin{itemize}
    \item \textbf{\textit{Step 1: Initialization with manually designed state.}}  
    The process begins by constructing a baseline state representation composed of manually selected environment variables. These variables typically encode low-level physical parameters and control indicators relevant to communication and mobility in LAWNs. However, such handcrafted states often lack the semantic richness and strategic context necessary for high-level reasoning, especially when domain expertise is limited.

    \item \textbf{\textit{Step 2: LLM-driven state and reward augmentation.}}  
    To address the limitations of manual state design, LLMs are leveraged to enhance both the representational and evaluative dimensions of the learning environment. Using tailored task prompts, the LLM can interpret the original state and autonomously generate semantically enriched features that reflect abstract but mission-relevant concepts, such as the angle between the LAP and jammer. In parallel, it formulates intrinsic reward signals aligned with high-level communication goals, enabling reinforcement learning agents to internalize strategic preferences without relying on manually coded heuristics.

    \item \textbf{\textit{Step 3: Integrated reinforcement learning with augmented representation.}}  
    The enhanced state and reward constructs are then integrated into the reinforcement learning pipeline. This augmented structure enables the agent to learn policies that go beyond immediate signal conditions, capturing broader mission context and long-term tradeoffs critical to secure communications. Note that our proposed framework can be applied to various existing RL methods without structural adjustments.
\end{itemize}
