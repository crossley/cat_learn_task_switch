Cognitive Control Over Learning: Creating, Clustering, and
Generalizing Task-Set Structure

Anne G. E. Collins and Michael J. Frank

> Understanding how they interact requires studying how
> cognitive control facilitates learning but also how learning
> provides the (potentially hidden) structure, such as
> abstract rules or task-sets, needed for cognitive control.

Yep.

> First, we develop a new context-task-set (C-TS) model,
> inspired by nonparametric Bayesian methods, specifying how
> the learner might infer hidden structure (hierarchical
> rules) and decide to reuse or create new structure in
> novel situations. 

Yep.

> Second, we develop a neurobiologically explicit network
> model to assess mechanisms of such structured learning in
> hierarchical frontal cortex and basal ganglia circuits.

Will be cool to see how they get the BG to do this.

> Third, this synergism yields predictions about the nature
> of human optimal and suboptimal choices and response times
> during learning and task-switching.

Seems pretty relevant. 

> These findings implicate a strong tendency to
> interactively engage cognitive control and learning,
> resulting in structured abstract representations that
> afford generalization opportunities and, thus, potentially
> long-term rather than short-term optimality.

Keep an eye on the type of learning modelled. What types of
learning are in scope for their model? The description here
sounds a lot like rule-based but not procedural learning.

Also keep an eye out for II versions of whatever experiment
design they come up with.

> Extensive task-switching literature has revealed the
> existence of task-set representations in both mind and
> brain (functional magnetic resonance imaging: Dosenbach et
> al., 2006; monkey electrophysiology: Sakai, 2008; etc.).
> Notably, these task-set representations are independent of
> the context in which they are valid (Reverberi, Görgen, &
> Haynes, 2011; Woolgar, Thompson, Bor, & Duncan, 2011) and
> even of the specific stimuli and actions to which they
> apply (Haynes et al., 2007) and are thus abstract latent
> constructs that constrain simpler choices.

Again sounds like rule-based learning.

> Conversely, the reinforcement learning (RL) literature has
> largely focused on how a single rule is learned and
> potentially adapted, in the form of a mapping between a
> set of stimuli and responses.

This framing is only rule-based if the number of stimuli is
low and they are highly discriminable.

> How do humans simultaneously learn (a) the simple
> stimulus–response associations that apply for a given
> task-set and (b) at the more abstract level, which of the
> candidate higher order task-set rules to select in a given
> context (or whether to build a new one)?

Well put.

> For example, subjects learned more efficiently when a
> simplifying rule-like structure was available in the set
> of stimulus–action associations to be learned (“policy
> abstraction”; Badre, Kayser, & D’Esposito, 2010). Collins
> and Koechlin (2012) showed that subjects build repertoires
> of task-sets and learn to discriminate between whether
> they should generalize one of the stored rules or learn a
> new one in a new temporal context.

NOTE: maybe relevent to cite.

> However, these studies did not address whether and how
> subjects spontaneously and simultaneously learn such rules
> and sets of rules when the learning problem does not in
> some way cue that organization. One might expect such
> structure building in part because it may afford a
> performance advantage for subsequent situations that
> permit generalization of learned knowledge.

This is a key aspect of their work.

> Thus, cognitive control may be necessary for hypothesis
> testing about current states that act as contexts for
> learning motor actions (e.g., treating the internally
> maintained state as if it was an observable stimulus in
> standard RL).

> Indeed, recent behavioral modeling studies have shown that
> subjects can learn hidden variables such as latent states
> relevant for action selection, as captured by Bayesian
> inference algorithms or approximations thereof (Collins &
> Koechlin, 2012; Frank & Badre, 2012; Gershman, Blei, &
> Niv, 2010; Redish, Jensen, Johnson, & Kurth-Nelson, 2007;
> Todd, Niv, & Cohen, 2008; Wilson & Niv, 2011).

Solid list of references.

> Some studies have shown that subjects even tend to infer
> hidden patterns in the data when they do not exist and
> afford no behavioral advantage (Yu & Cohen, 2009) or when
> it is detrimental to do so (Gaissmaier & Schooler, 2008;
> Lewandowsky & Kirsner, 2000). Thus, humans may exhibit a
> bias to use more complex strategies even when they ‘are
> not useful, potentially because these strategies are
> beneficial in many real-life situations.

Yet in category learning the 1D rule is so persuasive and
tempting.

> Computational Models of Reinforcement Learning, Category
> Learning, and Cognitive Control

Well there is a section labelled category learning. Hmm.

> This theoretical framework has been successfully leveraged
> in the domain of category learning (e.g., Gershman & Blei,
> 2012; Gershman et al., 2010; Sanborn, Griffiths, &
> Navarro, 2006, 2010), where latent category clusters are
> created that allow principled grouping of perceptual
> inputs to support generalization of learned knowledge,
> even potentially inferring simultaneously more than one
> possible relevant structure for categorization (Shafto,
> Kemp, Mansinghka, & Tenenbaum, 2011).

What kind of category learning do these studeis look at?

> Here, we take some inspiration from these models of
> perceptual clustering and extend them to support
> clustering of more abstract task-set states that then
> serve to contextualize lower level action selection.

Yep.

> In particular, we show that a multiple-loop
> corticostriatal gating network using RL can implement the
> requisite computations to allow task-sets to be created or
> reused. The explicit nature of the mechanisms in this
> model allows us to derive predictions regarding the
> effects of biological manipulations and disorders on
> structured learning and cognitive control. Because it is a
> process model, it also affords predictions about the
> dynamics of action selection within a trial and, hence,
> response times.

Sounds pretty relevant and awesome.

> Thus, for each new context, the probability of creating a
> new task-set is proportional to , and the probability of
> reusing one of the known task-sets is proportional to the
> popularity of that task-set across multiple other
> contexts.

They changed this more recently I think but this is the same
as the Gershman work.

> This task included 16 different contexts and three stimuli,
> presented in interleaved fashion. Six actions were available
> to the agent.

NOTE: simulation 1

> These simulations include two successive learning phases,
> which for convenience we label training and test phase
> (see Figure 2, bottom left). The training phase involved
> just two contexts (C1 and C2), two stimuli (S1 and S2),
> and four available actions.

NOTE: simulation 2

> These circuits are arranged hierarchically, with two main
> “diagonal” frontal-BG connections from the higher to the
> lower loop striatum and subthalamic nucleus. The
> consequences are that (a) motor actions to be considered
> as viable are constrained by task-set selection and (b)
> conflict at the level of task-set selection leads to
> delayed responding in the motor loop, preventing premature
> action selection until the valid task-set is identified.
> As we show below, this mechanism not only influences local
> within-trial reaction times but also renders learning more
> efficient across trials by effectively expanding the state
> space for motor action selection and thereby reducing
> interference between stimulus–response mappings across
> task-sets.

> These dopaminergic prediction error signals transiently
> modify go and no-go activation states in opposite
> directions, and these activation changes are associated
> with activity-dependent plasticity, such that synaptic
> strengths from corticostriatal projections to active go
> neurons are increased during positive prediction errors,
> while those to no-go neurons are decreased, and vice versa
> for negative prediction errors.

Well enough I suppose but it's interesting to see the
contrast to Greg's Darp32 pathway style of doing things.

> As in the Bayesian C-TS model, we do not consider here the
> learning of which dimension should act as context or
> stimulus but assume they are given as such to the model
> and investigate the consequential effects on learning.

Okay.

> Thus, only the context (e.g., color) part of the sensory
> input projects to PFC, whereas the stimulus (e.g., shape)
> projects to posterior visual cortex. The stimulus
> representation in parietal cortex (PC) is then
> contextualized by top-down projections from PFC. Weights
> linking the shape stimulus inputs to PC are predefined and
> organized (top half of layer reflects Shape 1 and bottom
> half Shape 2). In contrast, projections linking color
> context inputs to PFC are fully and randomly connected
> with all PFC stripes, such that PFC representations are
> not simply frontal “copies” of these contexts; rather,
> they have (initially) no intrinsic meaning but, as we
> shall see, come to represent abstract states that
> contextualize action selection in the lower motor action
> selection loop.

Good model detail.

> When a PFC stripe is gated in response to the color
> context, this PFC representation is then multiplexed with
> the input shape stimulus in the PC, such that PC units
> contain distinct representations for the same sensory
> stimulus in the context of distinct (abstract) PFC
> representations (Reverberi et al., 2011). Specifically,
> while the entire top half (all three columns) of the PC
> layer represents Shape 1 and the bottom half Shape 2, once
> a given PFC stripe is gated, it provides preferential
> support to only one column of PC units (and the others are
> suppressed due to lateral inhibition).
