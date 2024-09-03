# Dreisbach & Haider, 2008, Psychological Research

> Goal-directed behavior requires the cognitive system to
> distinguish between relevant and irrelevant information.
> The authors show that task sets help to shield the system
> from irrelevant information. Participants had to respond
> to eight diVerent colored word stimuli under diVerent
> instruction conditions. They either had to learn the
> stimulus–response mappings (SR condition), to use one task
> set (1 TS condition) or to use two diVerent task sets (2
> TS condition). In the 2 TS and the SR conditions,
> participants showed response repetition eVects
> (interaction of color repetition £ response repetition),
> indicating that participants processed the color of the
> words. Importantly, the 1 TS condition did not show such
> an interaction. Overall, the results provide evidence for
> the shielding function of task sets. This beneWt turns
> into costs in classical task switching paradigms. From
> this perspective, switch costs can be interpreted as the
> consequence of successful shielding on the previous task.

- NOTE: Fair enough. Instructions matter. Difference between
  task-set switching and learning CSR mappings. So how do
  our instructions and procedure map onto these?

# Forrest et al., 2014, proc ann meet cog sci

> Both the model and CSR group produced small switch costs,
> mostly due to incongruent stimuli, and large congruency
> effects that reduced with practice.

- NOTE: Good reminder that perhaps we should look at
  congruency effects. So, what is a congruency effect? I
  think it's a difference in performance when consecutive
  stimuli require the same response versus when they do not.
  I don't think we examined this in out earlier switch paper
  either, which would provide a chance to look at congruency
  absent task switching (because of the training blocks).
  There might be a quick little paper here.

- Otherwise pretty simple little paper and same comments as
  for the other Forrest paper so moving on now.

# Forrest et al., 2014 JEP:LMC; 

> Task-cuing experiments are usually intended to explore
> control of task set. But when small stimulus sets are
> used, they plausibly afford learning of the response
> associated with a combination of cue and stimulus, without
> reference to tasks. In 3 experiments we presented the
> typical trials of a task-cuing experiment: a cue (colored
> shape) followed, after a short or long interval, by a
> digit to which 1 of 2 responses was required. In a tasks
> condition, participants were (as usual) directed to
> interpret the cue as an instruction to perform either an
> odd/even or a high/low classification task. In a cue
> stimulus ¡ response (CSR) condition, to induce learning of
> mappings between cue–stimulus compound and response,
> participants were, in Experiment 1, given standard task
> instructions and additionally encouraged to learn the CSR
> mappings; in Experiment 2, informed of all the CSR
> mappings and asked to learn them, without standard task
> instructions; in Experiment 3, required to learn the
> mappings by trial and error. The effects of a task switch,
> response congruence, preparation, and transfer to a new
> set of stimuli differed substantially between the
> conditions in ways indicative of classification according
> to task rules in the tasks condition, and retrieval of
> responses specific to stimulus– cue combinations in the
> CSR conditions. Qualitative features of the latter could
> be captured by an associative learning network. Hence
> associatively based compound retrieval can serve as the
> basis for performance with a small stimulus set. But when
> organization by tasks is apparent, control via task set
> selection is the natural and efficient strategy.

- NOTE: The experiments and interpretation seem simple
  enough. The question is how it relates to the current
  switch revision. I suppose we should first figure out if
  our design is instructed or trial-and-error as conceived
  by this paper. Then add it to the discussion along with
  the other task switch factors.


# Musslick and Cohen, 2021, TiCS

> Limitations in the capability to multitask can be
> explained by representation sharing between tasks.
> Computational modeling suggests that neural systems trade
> the benefits of shared representation for rapid learning
> and generalization (a mechanism increasingly exploited in
> machine learning) against constraints on multitasking
> performance. Experimental studies posit a trade-off
> between cognitive stability and cognitive flexibility.
> Computational analyses of this trade-off suggest that
> adaptations to high demands for flexibility limit the
> amount of control that can be allocated toasingletask.

- shared representation enables rapid learning but hurts
  multitasking?

- So far seems very much in line with the NSW multitasking
  person.

- NOTE: I think the link to the IIII and RBRB full reversal
  ought to be easier (whatever easier means) comment is
  already served just by reading the top level summary /
  abstract. I suppose the idea would either be that
  switching between similar representation may be more or
  less difficult than switching between dissimilar
  representations. It could also be that with a full
  reversal you only need one representation so that may
  confer some ease in learning. It's all pretty vague but I
  think that's what they're thinking.

- NOTE: Moving on for now.

# Flesch et al., 2022, Neuron

> How do neural populations code for multiple, potentially
> conflicting tasks? Here we used computational simulations
> involving neural networks to define ‘‘lazy’’ and ‘‘rich’’
> coding solutions to this context-dependent decision-making
> problem, which trade off learning speed for robustness.
> During lazy learning the input dimensionality is expanded by
> random projections to the network hidden layer, whereas in
> rich learning hidden units acquire structured
> representations that privilege relevant over irrelevant
> features. For contextdependent decision-making, one rich
> solution is to project task representations onto
> low-dimensional and orthogonal manifolds. Using behavioral
> testing and neuroimaging in humans and analysis of neural
> signals from macaque prefrontal cortex, we report evidence
> for neural coding patterns in biological brains whose
> dimensionality and neural geometry are consistent with the
> rich learning regime.

- Is lazy basically a sparse ESN?

> For example, we can switch nimbly between sequential tasks
> that require distinct responses to the same input data, as
> when alternately judging fruit by shape or size and
> friends by gender or age (Roy et al., 2010; Mante et al.,
> 2013; Saez et al., 2015; Takagi et al., 2020).

- Good examples.

> One recently popular theory proposes that stimulus and
> context signals are projected into a high-dimensional
> neural code, permitting linear decoding of exhaustive
> combinations of task variables (Fusi et al., 2016).

- REF.

> An alternative theory states that neural representations
> are mixed selective but structured on a low-dimensional
> and task-specific manifold (Ganguli et al., 2008; Sadtler
> et al., 2014; Gao and Ganguli, 2015; Chaudhuri et al.,
> 2019; Cueva et al., 2020) where correlated patterns of
> firing confer robustness on the population code (Zohary et
> al., 1994).

- Okay so two theories. All good if but a little vague.

> The question of whether neural codes are task agnostic or
> task specific speaks to core problems in neural theory
> with widespread implications for understanding the coding
> properties of neurons and neural populations (Yuste, 2015;
> Saxena and Cunningham, 2019).

- Nicely said.

> An emergent theme in machine learning research is that
> neural networks can solve nonlinear problems in two
> distinct ways, dubbed the ‘‘lazy’’ and ‘‘rich’’ regimes,
> which, respectively, give rise to high- and
> low-dimensional representational patterns in the network
> hidden units (Chizat et al., 2018; Jacot et al., 2018;
> Arora et al., 2019; Lee et al., 2019; Woodworth et al.,
> 2020). In the lazy regime, which occurs when weights in
> the hidden layers are initialized with draws from a
> distribution with high variance, the dimensionality of the
> input signals is expanded via random projections to the
> hidden layer such that learning is mostly confined to the
> readout weights. In the rich regime, which occurs under
> low initial variance, the hidden units instead learn
> highly structured representations that are tailored to the
> task demands (Saxe et al., 2019; Geiger et al., 2020;
> Woodworth et al., 2020; Paccolat et al., 2021).

- REFS.

> We used neural network simulations to characterize the
> nature of these solutions for a canonical
> context-dependent decision-making setting and employed
> representational similarity analysis to explore their
> neural geometry. Subsequently, we compared these
> observations to BOLD (blood-oxygen-level-dependent) data
> recorded when humans performed an equivalent task and to
> neural signals previously recorded from macaque prefrontal
> cortex (PFC) during context-dependent decisions (Mante et
> al., 2013). In humans, we found that dorsal portions of
> the PFC and posterior parietal cortex share a neural
> geometry and dimensionality with networks that are trained
> in the rich regime.

- NOTE: Perhaps a contribution to be made here is in
  time-resolved version of these analyses / insights? That
  is, when do human representations match one model or the
  other.

- NOTE: Could be lots of relevant stuff here for the
  mechanisms of context paper I want to write.

- NOTE: It seems like an awesome paper that I will want to
  take inspiration from for future work but I'm not seeing
  a clear link to the current switch revisions. So stopping
  for now.

# Flesch et al., 2018, PNAS
> By contrast, standard supervised deep neural networks
> trained on the same tasks suffered catastrophic forgetting
> under blocked training, due to representational
> interference in the deeper layers. However, augmenting
> deep networks with an unsupervised generative model that
> allowed it to first learn a good embedding of the stimulus
> space (similar to that observed in humans) reduced
> catastrophic forgetting under blocked training. Building
> artificial agents that first learn a model of the world
> may be one promising route to solving continual task
> performance in artificial intelligence research.

- Sounds super cool.

> One theory explains continual learning by combining
> insights from neural network research and systems
> neurobiology, arguing that hippocampal-dependent
> mechanisms intersperse ongoing experiences with recalled
> memories of past training samples, allowing replay of
> remembered states among real ones (7, 8). This process
> serves to decorrelate inputs in time and avoids
> catastrophic interference in neural networks by preventing
> successive overfitting to each task in turn. Indeed,
> allowing neural networks to store and “replay” memories
> from an episodic buffer can accelerate training in
> temporally autocorrelated environments, such as in video
> games, where one objective is pursued for a prolonged
> period before the task changes (5, 9).

- Cool.

> Similar results have been reported in human category
> learning, with several studies reporting an advantage for
> mixing exemplars from different categories, rather than
> blocking one category at a time (13, 14).

- REFS

- Not clear if these and the other references mentioned in
  this paragraph refer to blocking / interleaving As and Bs
  or if they refer to whole category structures. 

> We taught human and artificial agents to classify
> naturalistic images of trees according to whether they
> were more or less leafy (task A) or more or less branchy
> (task B), drawing trial-unique exemplars from a uniform
> bidimensional space of leafiness and branchiness (Fig.
> 1A).

- Whatever method they end up using here it seems like it
  may play nicely with Ben's unstructured data.

> This benefit was greatest for those individuals whose
> prior representation of the stimulus space (as measured by
> preexperimental similarity judgments among exemplars)
> organized the stimuli along the cardinal task axes of
> leafiness and branchiness.

- Well may not work so well for unstructured categories
  after all since we cannot define the dimensions nicely
  like this.

> Surprisingly, we even found evidence for the protective
> effect of blocked learning after rotating the category
> boundaries such that rules were no longer verbalizable.

- Interesting. Of course it could be simply that branchiness
  and leafiness are verbalisable.

> These findings suggest that temporally autocorrelated
> training objectives encourage humans to factorize complex
> tasks into orthogonal subcomponents that can be
> represented without mutual interference.

- It will be fun to think about how this fits into the
  Collins  stuff on task set creation etc.

> Subsequently, we trained a deep neural network to solve
> the same problem, learning by trial and error from image
> pixels alone.

- Will be cool to see trial-by-trial supervised learning in
  a deep net. I suppose this is learning in real time.

> As expected, a standard supervised deep network exhibited
> catastrophic forgetting under blocked training, and we
> used multivariate analysis of network activations to
> pinpoint the source of interference to the deeper network
> layer.

- Sounds like a cool method.

> Using an approach related to representational similarity
> analysis (RSA) (21, 22), we correlated human choices
> matrices for each of the two tasks with those predicted by
> two different models (Fig. 2D). The first used the single
> best possible linear boundary in tree space (linear
> model), and the second used two boundaries that cleaved
> different compressions of the tree space optimally
> according to the relevant dimension (factorized model).

- Probably okay but also why not use DBM?

> After appropriately penalizing for model complexity, we
> subjected model fits to Bayesian model selection at the
> random effects level (23–25).

- REFS. Cool method?

- NOTE: The RSA approach using explicit similarity
  judgements is kinda cool. I suppose it suffers from the
  same weakness as verbal reports in that it is measured
  after the fact. It does make me wonder about designs in
  where we include a similarity judgement pre and post
  category learning training.

> Next, for comparison with humans, we trained deep
> artificial neural networks to perform the task. In
> experiment 3, convolutional neural networks (CNNs) were
> separately trained on the cardinal and diagonal tasks
> under either blocked or interleaved training conditions,
> and classification performance was periodically evaluated
> with a held-out test set for which no supervision was
> administered.

- It's actually pretty wild this is in PNAS. Not a bad paper
  but not really sharp either. Here, the obvious issue is
  that the network training is supervised but the human
  training is not / or is reinforcement based. So the
  blocked vs interleaved question is muddled.

> On each “trial,” the networks received images of
> task-specific gardens onto which trees were superimposed
> as input (analogous to the content of the stimulus
> presentation period in the human experiments) and were
> optimized (during training) with a supervision signal
> designed to match the reward given to humans

- Hmm. Maybe the supervised learning signal is funky in some
  specific way.

- NOTE: Okay. Cool paper but stopping for now. I think the
  only real aspect related to the revision of our current
  switch paper is that we previously used blocked training
  and we currently use interleaved training. This PNAS paper
  suggests the interleaved training should be shittier. We
  show that interleaved training is fine provided you have 4
  resp. Something like that.

# Collins, 2017, J Cog Neuro

- Some behaviour and modelling in the same vein as all the
  rest here.
- skipping for now.

# Collins & Frank, 2014, J Neuro
- Neural / EEG spin on this line of studies.
- Skipping for now.

- NOTE: It seems like one thing we could add to this
  literature is some investigation of how this stuff works
  with II categories or II / RB dissociation.

# Collins & Frank, 2016, Cognition
- Neural / EEG spin on this line of studies.
- Skipping for now.

# Collins & Frank, 2016, PLoS Comp Biol
> Across four independent data-sets, we show that subjects
> create rule structures that afford motor clustering,
> preferring structures in which adjacent motor actions are
> valid within each task-set. In a fifth data-set using
> instructed rules, this bias was strong enough to
> counteract the well-known task switch-cost when
> instructions were incongruent with motor clustering.

- Cool.

> Specifically, cognitive control relies on a cascading
> hierarchical structure, where at a more abstract level,
> subjects choose task-sets appropriate to the context,
> which then constrain our action choices in response to
> lower-level, less abstract stimuli [3,4]. Thus, we use the
> term “context” to refer to those features that cue
> abstract task-set, and the term “stimulus” to refer to
> features that cue the appropriate response conditioned on
> the selected task-set.

- This is a nice way to split things but I've been wondering
  how it all lines up with Gurney at al models. How is
  channel salience determined?

> In the study of executive functions, researchers tend to
> focus on discretized aspects of decision making–often
> assuming that perception systems have transformed complex,
> multidimensional sensory signals into reduced, discrete
> stimuli (e.g., a red circle), and that given this percept,
> the executive system selects among a few discrete options
> (e.g., left vs. right), which are then implemented by the
> motor system. However, the field of embodied or grounded
> cognition [5–7] offers strong hints that this model is
> over-simplified, emphasizing instead that executive
> functions evolved for the control of action in continuous
> time [5], and are thus scaffolded on existing sensorimotor
> processing systems.

- This seems like a great intro to reachy cat projects.
- The paragraph goes on and is full o freally good stuff.

> Here, we propose that an intrinsic constraint of motor
> action representations strongly influences how we create
> representations of abstract hierarchical rule structures,
> both during learning and while applying instructed rules.

- Sounds like it might offer some insight into 2-resp vs
  4-resp conditions.

- NOTE: seems like some of the transfer experiments they
  seems to do in their studies would be feasible to extend
  to bees et al.

> Here, we investigated whether motor patterns provide an
> additional constraint on imposing hierarchical cognitive
> rule structures. To this effect, we take advantage of
> motor representational structure in motor cortex. Finger
> movement representations in M1, while globally
> somatotopic, are widely overlapping for neighboring
> fingers [23,24], reflecting the natural statistics of hand
> movement structure [24]. This representational constraint
> may lead to pre-activation of neural networks representing
> frequently co-activated fingers (motor synergies [25,26]),
> thus facilitating further choices with those fingers. For
> example, cues that allow for mot

- So does this say 2-resp vs 4-resp is expected or
  surprising? Does it say we should see different results
  depending on hand / finger assignments?

> This would then lead to a natural constraint on the
> creation of abstract rule structure: subjects should treat
> those feature dimensions that afford motor clustering as
> lower order stimulus (allowing clustering within rules),
> and those that afford less motor clustering as higher
> order context.

- relevant but hard to parse this stuff.

- NOTE: Moving on for now.

# Collins & Frank, 2013, Psych Rev

- Definitely seems super relevant but not clear at the
  moment if they will look at switching during learning.

> How do humans simultaneously learn (a) the simple
> stimulus–response associations that apply for a given
> task-set and (b) at the more abstract level, which of the
> candidate higher order task-set rules to select in a given
> context (or whether to build a new one)?

- Seeming more like they will be stealing my thunder
  shortly.

- On the other hand it now isn't clear how this model will
  capture switch costs etc. It seems more about the Bayesian
  creation of contexts. No?

- NOTE: It actually seems like their model ought to be able
  to learn under 2-response and 4-response option
  conditions. Or perhaps they are not looking at instances
  of pure SR reversal.

- NOTE: In any case it does seem like this model would
  belong in a network mechanisms of context review paper.

- Sounds like it's going to be very Bayesian / math like the
  Gershman stuff.

> This theoretical framework has been successfully leveraged
> in the domain of category learning (e.g., Gershman & Blei,
> 2012; Gershman et al., 2010; Sanborn, Griffiths, &
> Navarro, 2006, 2010), where latent category clusters are
> created that allow principled grouping of perceptual
> inputs to support generalization of learned knowledge,
> even potentially inferring simultaneously more than one
> possible relevant structure for categorization (Shafto,
> Kemp, Mansinghka, & Tenenbaum, 2011). Furthermore,
> although optimal inference is too computationally
> demanding and has high memory cost, reasonable
> approximations have been adapted to account for human
> behavior (Anderson, 1991; Sanborn et al., 2010).

- REFS

> In brief, the perceptual category learning literature
> typically focuses on learning categories based on
> similarity between multidimensional visual exemplars. In
> contrast, useful clustering of contexts for defining
> task-sets relies not on their perceptual similarity but
> rather in their linking to similar stimulus–action–
> outcome contingencies (see Figure 1), only one “dimension”
> of which is observable in any given trial. We thus extend
> similarity-based category learning from the mostly
> observable perceptual state space to an abstract, mostly
> hidden but partially observable rule space.

- Cool.

> In the vast majority, learning problems are modeled with
> RL algorithms that assume perfect knowledge of the state,
> although some recent models include state uncertainty or
> learning about problem structure in specific circumstances
> (Acuña & Schrater, 2010; Botvinick, 2008; Collins &
> Koechlin, 2012; Frank & Badre, 2012; Green, Benson,
> Kersten, & Schrater, 2010; Kruschke, 2008; Nassar, Wilson,
> Heasly, & Gold, 2010; Wilson & Niv, 2011).

- REFS

> Recently, Frank and Badre (2012) proposed a hierarchical
> extension of this gating architecture for increasing
> efficiency and reducing conflict when learning multiple
> tasks. Noting the similarity between learning to choose a
> higher order rule and learning to select an action within
> a rule, they implemented these mechanisms in parallel
> gating loops, with hierarchical influence of one loop over
> another.

- REF.

> The inclusion of four actions (as opposed to two, which
> are overwhelmingly used in the task-switching literature,
> but see Meiran & Daichman, 2005) allows us to analyze not
> only accuracy but also the different types of errors that
> can be made.

- REF.
- But I don't think this is quite the same context as ours.

> Notably, in the model proposed below, there are two such
> circuits, with one learning to gate an abstract task-set
> (and to cluster together contexts indicative of the same
> task-set) and the other learning to gate a motor response
> conditioned on the selected task-set and the perceptual
> stimulus. These circuits are arranged hierarchically, with
> two main “diagonal” frontal-BG connections from the higher
> to the lower loop striatum and subthalamic nucleus.

- This sounds a lot like the two-stage model.

> This functionality provides a dynamic regulation of the
> model’s decision threshold as a function of response
> conflict (Ratcliff & Frank, 2012), such that more time is
> taken to accumulate evidence among noisy corticostriatal
> signals to prevent impulsive responding and to settle on a
> more optimal response.

- This is STN stuff.

> Below, we describe a novel extension of this mechanism to
> multiple frontal-BG circuits, where conflict at the higher
> level (e.g., during task-switching) changes motor response
> dynamics.

- Hmm. Will be cool to see how the STN plays into switching
  / context stuff.

- NOTE: The role of STN in setting response thresholds and
  its role here in context / cognitive control seems like it
  ought to be testable in DBS patients. It also seems like
  Frank et al would be all over this.

- NOTE: seems like this is also basically my multiple loop
  instantiation of RB / II category learning. Just need to
  tweak some PFC things or maybe not even that. I could
  probably just implement it as is and see how it does on
  the classic dissociations: button switch, feedback delay,
  dual-task etc.

- NOTE: I think this is learning and switching / cognitive
  control at the same time. The situation is a bit different
  though. So far they are only explore relatively few
  stimuli etc. Maybe some other differences. Bit we for sure
  will need to change our framing.

- NOTE: I think a big difference in framing is that this
  paper is about learning / inferring task sets, whereas our
  study explicitly à imposes task sets. 

> Other category learning models focus on the division of
> labor between BG and PFC in incremental procedural
> learning versus rule-based learning but do not consider
> rule clustering. In particular, the COVIS model (Ashby,
> Alfonso-Reese, Turken, & Waldron, 1998) involves a PFC
> component that learns simple rules based on hypothesis
> testing. However, COVIS rules are based on perceptual
> similarity and focus on generalization across stimuli
> within rules rather than generalization of rules across
> unrelated contexts. Thus, although COVIS could learn to
> solve the tasks we study (in particular, the flat model is
> like a conjunctive rule), it would not predict transfer of
> the type we observed here to other contexts. Other models
> rely on different systems (such as exemplar,
> within-category clusters, and attentional learning) to
> allow learning of rules less dependent on similarity
> (Hahn, Prat-Sala, Pothos, & Brumby, 2010; Kruschke, 2011;
> Love, Medin, & Gureckis, 2004). Again, these models allow
> generalization of rules not across different contexts but
> only potentially across new stimuli within the rules.

- Good.


# gilbert & shallice (2002, cogn psychol)

> The distinction between individual S-R mappings and task
> sets, composed of all of the individual S-R mappings which
> are required to carry out an experimental task, is
> crucial.  As we shall see, an important issue concerns the
> degree to which task switching should be understood in
> terms of processes occurring at the level of discrete S-R
> mappings rather than those which occur at the level of the
> task set (see also Monsell, Taylor, & Murphy, 2001).

- Nice.

> The task demand units receive an additional ‘‘top-down
> control input,’’ which specifies which task the model
> should perform. The word and color output units are
> interconnected, so that congruent word and color response
> units (e.g., the two ‘‘red’’ units) have reciprocal
> positive connections and incongruent pairs of units (e.g.,
> word ‘‘red’’/color green) have reciprocal negative
> connections.

- Seems pretty convenient to simply put excitation and
  inhibition wherever suits you like this. No?

> Finally, there are lateral inhibitory connections between
> all units within the word output ‘‘module’’ (i.e., set of
> word output units), color output module and task demand
> module. This encourages the network to settle into stable
> states with no more than one unit active in each module.

- This seems close to our model.

- It is difficult to map precisely but it seems like their
  output units are most similar to our category label units.

> Although the model is based on the earlier models of
> Cohen, Dunbar, and McClelland (1990) and Cohen and Huston
> (1994), its architecture differs from them in two main
> respects. First, there are three possible words and colors
> (red, green, and blue) in the present model as opposed to
> just two in the earlier models. This was chosen because
> the Cohen et al. (1990) model has been criticized for
> failing to capture differences between word reading and
> color naming when the set size is increased beyond two
> (Kanne, Balota, Spieler, & Faust, 1998; but see Cohen,
> Usher, & McClelland, 1998, for a reply). A second
> difference is that, unlike our model, the earlier models
> included a ‘‘winner-take-all’’ response layer. In the
> Cohen et al. (1990) model, the units corresponding to the
> word and color output units in the present model sent
> inputs into a pair of response units. Thus the
> word-reading ‘‘red’’ output unit and the color-naming
> ‘‘red’’ output unit sent activation to a single ‘‘red’’
> response unit. Evidence was collected from these response
> units in order to determine when each trial should end.
> These additional units are unnecessary in the present
> model: since we have interconnected the word and color
> output units there is a beneficial effect of activating
> congruent output units and a detrimental effect of
> activating incongruent output units. This plays a role in
> our model similar to the convergent inputs into the
> response units of the Cohen et al. (1990) and Cohen and
> Huston (1994) models.

- There are some useful hints in here to pull out.
- Good old references as well.

NOTE: moving on to more recent work now since this is all
feeling pretty familiar and perhaps too far into the weeds.


# brown, reynolds & braver (2006, cogn psychol)
> we suggest that cognitive control may include, among other
> things, two distinct processes. incongruent stimuli may
> drive top-down facilitation of task-relevant responses to
> bias performance toward exploitation vs. exploration. task
> or response switches may generally slow responses to bias
> toward accuracy vs. speed and exploration vs. exploitation.
> behavioral results from a task switching study demonstrate
> these two distinct processes as revealed by higher-order
> sequential effects. a computational model implements the two
> conflict-control mechanisms, which allow it to capture many
> complex and novel sequential effects. lesion studies with
> the model demonstrate that the model is unable to capture
> these effects without the conflict-control loops and show
> how each monitoring component modulates cognitive control.
> the results suggest numerous testable predictions regarding
> the neural substrates of cognitive control.

1. top-down facilitation of task-relevant responses
2. task or response switches may generally slow responses

> thus, we postulate that there are multiple
> conflict-control loop mechanisms in the brain that are
> each associated with regulating adjustment in specific
> forms of cognitive control.

- will be cool to see what these are exactly.

> in our model (fig. 7a), a task cue activates a persistent
> representation of task set. recurrent excitation in this
> task set layer allows for stable recirculation and active
> maintenance of the current task set. the pattern of
> activity represents the task set, and this activity in
> turn biases the transformation of target signals into
> movement commands according to the current task set. the
> model defines movement (such as a button press) as a
> behavioral action initiated when a corresponding model
> output layer cell’s activity reaches a fixed threshold
> (hanes & schall, 1996). we define a hidden or ‘‘plan’’
> layer between the target stimuli input and the output
> layer cells, where the movement is specified prior to its
> execution by the output layer cells. specifically, each
> plan layer cell responds to a unique combination of
> signals from task set representations and
> target-stimuli-representing input layer cells, and in turn
> drives the response. thus, the task-set layer activity
> pattern ensures execution of the task-appropriate response
> to the target.

- sounds pretty much like our model / the standard model  so
  far.

- but the list goes on and gets fairly opaque...

- one takeaway may be that our model brings something novel
  but only adds to an already multifaceted list of cognitive
  control processes.

- that is, we may need a section that briefly covers all the
  things our model does not include that this model does
  include.

> The network can be conceptualized in terms of two major
> divisions: (1) a network that can accomplish
> task-switching; and (2) a supervisory control system that
> interacts with this task-switching network.

1. INC loop (incongruency detection)
2. CH loop (change detection)

> Within the task-switching network, the architecture
> consisted of five layers of units: a target input layer, a
> cue input layer, a plan (or hidden) layer, a task-set
> layer and a response output layer.

Task switch:
- target input layer
- cue input lyer
- motor plan layer
- task-set layer
- motor output / response layer

> Finally, there was strong lateral inhibition among units
> in the task-set layer such that the presentation each new
> cue caused an updating of the relevant task-set activity.

> Two sets of connections were specifically affected by
> experience: the connection from the hidden to task-set
> layer and the connection from hidden to output layer. In
> both sets of connections the weight changes corresponded
> to a basic associative or Hebbian (Hebb, 1949) learning
> rule, such that co-activation of a sending and receiving
> unit led to a strengthening of the connection between
> them, in proportion to the magnitude of co-activation. The
> effects of changing the hidden layer to task-set
> connection implemented a form of associative TSI, since
> targets that have previously been associated with a
> specific task-set in a previous trial will more strongly
> activate this task-set when the targets are activated in a
> current trial (even if the previous task-set is no longer
> appropriate). The connections from the hidden to output
> layer implement a form of response priming, such that a
> response activated on a previous trial will have a
> strengthened connection to the hidden units that
> correspond to the stimulus features that activated the
> response. For both sets of adaptive connections the
> priming effect was time-dependent such that there was a
> passive exponential decay of any strengthening back to
> baseline values with a relatively short time constant
> (i.e., with a half-life of several seconds).

This is a pretty cool feature.


