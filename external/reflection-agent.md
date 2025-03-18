Language Agent Tree Search
--------------------------

**Links: (**[**Python**](https://github.com/langchain-ai/langgraph/blob/main/examples/lats/lats.ipynb?ref=blog.langchain.dev)**,** [**Youtube**](https://youtu.be/v5ymBTXNqtk?feature=shared&t=625&ref=blog.langchain.dev)**)**

[Language Agent Tree Search](https://arxiv.org/abs/2310.04406?ref=blog.langchain.dev) (LATS), by Zhou, et. al, is a general LLM agent search algorithm that combines reflection/evaluation and search (specifically Monte-Carlo trees search) to achieve better overall task performance compared to similar techniques like ReACT, Reflexion, or even Tree of Thoughts. It adopts a standard reinforcement learning (RL) task framing, replacing the RL agents, value functions, and optimizer all with calls to an LLM. This is meant to help the agent adapt and problem solve for complex tasks, avoiding getting stuck in repetitive loops.

The search process is outlined in the diagram below:

![](https://blog.langchain.dev/content/images/2024/02/lats.png)

The search has four main steps:

1.  **Select**: pick the best next actions based on the aggregate rewards from step (2) below. Either respond (if a solution is found or the max search depth is reached) or continue searching.
2.  **Expand and simulate:** generate N (5 in our case) potential actions to take and execute them in parallel.
3.  **Reflect + evaluate**: observe the outcomes of these actions and score the decisions based on reflection (and possibly external feedback)
4.  **Backpropagate**: update the scores of the root trajectories based on the outcomes.

If the agent has a tight feedback loop (through high quality environment rewards or reliable reflection scores), the search is able to accurately distinguish between different action trajectories and pick the best path. The final trajectory can then be saved to external memory (or used for model fine-tuning) to improve the model in the future.

The "selection" step picks the node with the highest upper confidence bound (UCT), which just balances the expected reward (the first term) with an incentive to explore new paths (the second term).

UCT\=valuevisits+cln⁡(parent.visits)visits UCT = \\frac{\\text{value}}{\\text{visits}} + c \\sqrt{\\frac{\\ln(\\text{parent.visits})}{\\text{visits}}}UCT\=visitsvalue​+cvisitsln(parent.visits)​​

Check out the [code](https://github.com/langchain-ai/langgraph/blob/main/examples/lats/lats.ipynb?ref=blog.langchain.dev) to see how it's implemented. In our LangGraph implementation, we put generation + reflection steps in a single node each, and check the tree state on each loop to see if the task is solved. The (abbreviated) graph definition looks something like below:

    from langgraph.graph import END, StateGraph
    
    class Node:
        def __init__(
            self,
            messages: List[BaseMessage],
            reflection: Reflection,
            parent: Optional[Node] = None,
        ):
            self.messages = messages
            self.parent = parent
            self.children = []
            self.value = 0
            self.visits = 0
        # Additional methods are defined here. Check the code for more!
    
    class TreeState(TypedDict):
        # The full tree
        root: Node
        # The original input
        input: str
    
    def should_loop(state: TreeState):
        """Determine whether to continue the tree search."""
        root = state["root"]
        if root.is_solved:
            return END
        if root.height > 5:
            return END
        return "expand"
    
    
    builder = StateGraph(TreeState)
    builder.add_node("start", generate_initial_response)
    builder.add_node("expand", expand)
    builder.set_entry_point("start")
    
    
    builder.add_conditional_edges(
        "start",
        # Either expand/rollout or finish
        should_loop,
    )
    builder.add_conditional_edges(
        "expand",
        # Either continue to rollout or finish
        should_loop,
    )
    
    graph = builder.compile()

LATS Graph

Once you've created the basic outline, it's easy to expand to other tasks! For instance, this technique would suit code generation tasks well, where you the agent can write explicit unit tests and score trajectories based on test quality.

LATS unifies the reasoning, planning, and reflection components of other agent architectures, such as Reflexion, Tree of Thoughts, and [plan-and-execute](https://blog.langchain.dev/planning-agents/) agents. LATS also from the backpropagation of reflective and environment-based feedback for an improved search process. While it can be sensitive to the reward scores, the general algorithm can be flexibly applied to a variety of tasks.

![](https://blog.langchain.dev/content/images/2024/02/image-17.png)

Comparison of LATS with other agent architecturese