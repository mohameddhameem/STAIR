"""Port of AgentTTS's search agent (github.com/FairyFali/AgentTTS, code_others/AgentTTS.py and agent/agent_class.py,
Apache License 2.0, (c) the AgentTTS authors). The classes below are their code with these changes only:
  * the planner LLM is a local model (Qwen2.5 72B through vLLM) instead of OpenAI's o3-mini;
  * the 'environment' looks configurations up in the table we measured on the 50 search questions (their CSV files from
    wandb sweeps are not public), with their own fuzzy matching (nearest sample counts, same models, within budget);
  * robustness: an unreadable LLM reply is asked again (up to 3 times) and malformed candidates are skipped instead of
    stopping the search; the random generators are seeded.
Prompts, budget formula, initial candidates, insight steps and the search loop are unchanged."""
import json
import random
import re
from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import pandas as pd  # noqa: F401

model_specs = {
    "Qwen/Qwen2.5-7B-Instruct-AWQ": {"M": 7_000_000_000, "L": 32, "D": 4_096},
    "Qwen/Qwen2.5-32B-Instruct-AWQ": {"M": 32_000_000_000, "L": 48, "D": 6_144},
    "Qwen/Qwen2.5-72B-Instruct-AWQ": {"M": 72_000_000_000, "L": 80, "D": 8_192},
    "meta-llama/Llama-3.2-3B-Instruct": {"M": 3_000_000_000, "L": 32, "D": 3_072},
    "meta-llama/Llama-3.1-8B-Instruct": {"M": 8_000_000_000, "L": 32, "D": 4_096},
    "meta-llama/Llama-3.1-70B-Instruct": {"M": 70_000_000_000, "L": 80, "D": 8_192},
    # added for the GPU smoke test (not in AgentTTS): published layer counts and hidden sizes
    "Qwen/Qwen2.5-0.5B-Instruct": {"M": 500_000_000, "L": 24, "D": 896},
    "Qwen/Qwen2.5-1.5B-Instruct": {"M": 1_500_000_000, "L": 28, "D": 1_536},
    "meta-llama/Llama-3.2-1B-Instruct": {"M": 1_000_000_000, "L": 16, "D": 2_048},
}

# module-level settings that AgentTTS.py keeps as globals of its __main__
no_insight1 = no_insight2 = no_insight3 = False
Np_list, Nd_list, model_space = [2048, 256], [128, 64], {}
_ENV = None
_LOG = print


class Task(object):
    def __init__(self, name, desc, budget, model_choices, pattern, sub_tasks, agent):
        self.name = name
        self.desc = desc
        self.budget = budget
        self.model_choices = model_choices
        self.pattern = pattern
        self.sub_tasks = sub_tasks
        self.agent = agent

    def get_tts_strategy(self, iterations, batch_size):
        return self.agent.run(iterations, batch_size, trace=True)

class AgentTTS(object):
    def __init__(self, task_name, task_desc, budget, model_choices, pattern, sub_tasks, Np_list, Nd_list, llm, example=None):
        self.task_name = task_name
        self.task_desc = task_desc
        self.budget = budget
        self.model_choices = model_choices
        self.pattern = pattern
        self.sub_tasks = sub_tasks
        self.Np_list = Np_list
        self.Nd_list = Nd_list
        self.archive = Archive(task_name)
        self.llm = llm
        self.example = example

        self.sub_tasks_dict = {}
        for subtask, model_list, Np, Nd in zip(sub_tasks, model_choices, Np_list, Nd_list):
            largest = 0
            for model in model_list:
                model_size = self.get_model_size(pattern, model)
                if model_size > largest:
                    largest = model_size
            self.sub_tasks_dict[subtask] = {}
            self.sub_tasks_dict[subtask]['largest_model_size'] = largest
            self.sub_tasks_dict[subtask]['Np'] = Np
            self.sub_tasks_dict[subtask]['Nd'] = Nd


        self.M_dict = {}
        for subtask, model_list in zip(sub_tasks, model_choices):
            largest = 0
            for model in model_list:
                model_size = self.get_model_size(pattern, model)
                if model_size > largest:
                    largest = model_size
            self.M_dict[subtask] = largest
        _LOG('M dict:', self.M_dict)

        assert len(model_choices) == len(sub_tasks)

    def run(self, iterations: int, batch_size: int, trace=True):
        main_metric = self.archive.environment.main_metric
        # step 1. init candidate, update history
        new_candidates = self.init_candidates()
        _LOG('### initial candidates:', new_candidates, '\n\n\n\n')
        new_candidates, scores = self.get_scores(new_candidates)  # scores are feedback
        _LOG('Log, scores for initial candidates:', scores)

        recent_added = 0
        for candidate, score in zip(new_candidates, scores):
            _LOG('### new candidate:', candidate)
            _LOG('### score:', score)
            cost = self.calculate_budget(candidate)
            if score and score[main_metric] > 0:
                self.update_archive_history(candidate, score, cost)
                recent_added += 1

        for iteration in range(iterations):
            _LOG(f"\n\nIteration {iteration + 1} - Generated candidates:\n{json.dumps(new_candidates, indent=2)}")

            # step 2. gen new experience, serve as further feedback
            if iteration == 0:
                init = True
            else:
                init = False
            new_experience = self.gen_new_experience(recent_added_samples=recent_added, init=init)
            self.update_experience(new_experience)

            # step 3. gen new candidates, update history
            new_candidates = self.gen_new_candidates(batch_size)
            new_candidates, scores = self.get_scores(new_candidates)  # list[dict]
            recent_added = 0
            for candidate, score in zip(new_candidates, scores):
                _LOG('### new candidate:', candidate)
                _LOG('### score:', score)
                cost = self.calculate_budget(candidate)
                if cost > self.budget:
                    _LOG('Budget exceeded')
                    continue

                if score and score[main_metric] > 0:
                    self.update_archive_history(candidate, score, cost)
                    recent_added += 1
                    _LOG(f"Score: {score[main_metric]:.2f} | Cost: {cost:.2f} | Params: {candidate}")
            _LOG("### Log, len(history):)", len(self.archive.history))
            # step 4. until the end

        # get the best score
        history = self.get_evaluation_results()
        main_metric = self.get_main_metric()
        results = []
        _LOG("history: ")
        for i, item in enumerate(history):
            params = item["params"]
            main_score = item["score"][main_metric]
            cost = item["cost"]
            results.append((i, params, main_score, cost))
            _LOG({'i': i, 'params': params, 'main_score': main_score, 'cost': cost})

        if trace:
            best_main_score = 0
            trace = []
            for step, item in enumerate(history):
                main_score = item["score"][main_metric]
                if main_score > best_main_score:
                    best_main_score = main_score
                    trace.append((step, best_main_score, item))
            return results, trace



        _LOG("\nFinal archive (sorted by score):")
        for i, params, score, cost in sorted(results, key=lambda x: -x[2]):
            _LOG(f"[{score:.2f}] {params} Used budget {cost}, index {i}")

        return results


    def update_experience(self, new_experience):
        self.archive.experience.append(new_experience)

    def get_scores(self, new_candidates):
        new_candidates, scores = self.archive.evaluate_batch(new_candidates, self.budget)
        return new_candidates, scores

    def update_archive_history(self, new_candidate, score, cost):
        self.archive.add_new_params(new_candidate, score, cost)

    def get_evaluation_results(self):
        return self.archive.history

    def get_main_metric(self):
        return self.archive.environment.main_metric

    @abstractmethod
    def gen_new_candidates(self, batch_size):
        '''
        使用archive里面的历史和上一次的经验来生成新的候选，生成候选的策略
        # (1) 突变采样数量，increase/decrease the samples
        # (2) 突变模型大小
        # (3) 交叉选择参数
        # (4) 随机生成新的参数组合
        设计内部函数，包括突变，交叉选择参数，和生成新的参数组合
        :param init:
        :return:
        '''
        pass

    @abstractmethod
    def gen_new_experience(self, recent_added_samples:int, init=False):
        pass

    @abstractmethod
    def init_candidates(self):
        pass

    def get_model_size(self, pattern, model_name):
        # r = re.findall(pattern, model_name)
        # if len(r) != 0:
        #     model_size = int(r[0])
        # else:
        #     model_size = 0
        model_size = model_specs[model_name]["M"]
        return model_size

    def extract_json(self, text: str) -> dict:
        try:
            start = text.find('{')
            end = text.rfind('}') + 1
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            return None

    def compute_budget(self, S, M_large, Np1, Nd1, M_small=3e9, Np2=128, Nd2=64):
        """
        Compute the FLOPs-based budget of generating S samples on a large model,
        normalized to the FLOPs of a small model's unit budget.

        Parameters:
        - S: number of samples generated by the large model
        - M_small: parameter count of the small model (e.g., 3B, ignore B)
        - M_large: parameter count of the large model (e.g., 70B)
        - Np2: prompt length for small model (unit config)
        - Nd2: decode length for large model (unit config)
        - Np1: prompt length for the new config
        - Nd1: decode length for the new config

        Returns:
        - budget: normalized compute budget
        """
        alpha = M_large / M_small
        beta1 = Np1 / Nd1
        beta2 = Np2 / Nd2  # normalize to same decode length as unit config
        beta3 = Np1 / Np2

        budget = beta2 * ((alpha * beta3 / beta1) * (beta1 + S) - 1)
        return budget

    def calculate_budget(self, params):
        try:
            return self._calculate_budget(params)
        except (KeyError, TypeError, ValueError):
            return float('inf')

    def _calculate_budget(self, params):
        def subunit_cost(subtask: str, model: str, samples: int) -> float:
            # M = self.M_dict[subtask]
            m = self.get_model_size(self.pattern, model)
            samples = samples
            Np = self.sub_tasks_dict[subtask]['Np']
            Nd = self.sub_tasks_dict[subtask]['Nd']

            budget = self.compute_budget(samples, m, Np, Nd)
            return budget

            # denominator = 4 * (M / m) - 3
            # return samples / denominator if denominator != 0 else float('inf')

        return sum(
            subunit_cost(t, p["model"], p["samples"])
            for t, p in params.items()
        )

    def check_budget(self, candidate: Dict):
        _LOG(candidate, 'budget:', self.calculate_budget(candidate))
        return self.calculate_budget(candidate) <= self.budget


class LLM(ABC):

    @abstractmethod
    def generate(self, prompt, system_prompt):
        pass


class Archive(object):
    def __init__(self, task_name):
        self.task_name = task_name
        self.history = []
        self.experience = []
        self.environment = _ENV

    def evaluate_batch(self, batch, budget):
        results = []
        new_parameters = []
        for parameter in batch:
            new_parameter, result = self.environment.run(parameter, budget)
            if new_parameter is not None and result is not None:
                results.append(result)
                new_parameters.append(new_parameter)
        return new_parameters, results

    def add_new_params(self, param, score, cost):
        self.history.append({"params": param, "score": score, "cost": cost})


class TableEnvironment(object):
    """Their Environment for retrieval QA, reading our measured table instead of their CSV."""

    def __init__(self, df, metrics=("Gen_EM", "Ret_F1"), main_metric="Gen_EM"):
        self.db = df
        self.metrics = list(metrics)
        self.main_metric = main_metric
        self.calls = []

    def run(self, params, budget):
        try:
            items, diff = self.fuzzy_match(params, budget)
        except (KeyError, TypeError, ValueError) as e:
            _LOG(f"skipped malformed candidate {params}: {type(e).__name__}")
            return None, None
        items = items.to_dict(orient="records")
        if diff > 5:
            return None, None

        def convert_row_to_params(row):
            return {"Retrieval": {"model": row["retrieve_model_name"], "samples": int(row["retrieve_samples"])},
                    "Question Answering": {"model": row["generate_model_name"], "samples": int(row["generate_samples"])}}
        if len(items) >= 1:
            out = convert_row_to_params(items[0]), {m: float(np.max([it[m] for it in items])) for m in self.metrics}
            self.calls.append({"asked": params, "looked_up": out[0], "score": out[1]})
            return out
        return None, None

    def fuzzy_match(self, params, budget=1000):
        retrieve_model = params['Retrieval']['model']
        retrieve_samples_target = params['Retrieval']['samples']
        generate_model = params['Question Answering']['model']
        generate_samples_target = params['Question Answering']['samples']
        filtered = self.db[
            (self.db['retrieve_model_name'] == retrieve_model) &
            (self.db['generate_model_name'] == generate_model) &
            (self.db['budget'] <= budget)
            ]
        if filtered.empty:
            items = filtered
            min_diff = -1
        else:
            ret_model_size = model_specs[retrieve_model]["M"]
            gen_model_size = model_specs[generate_model]["M"]
            d1 = 72e9/ret_model_size * 4 - 3
            d2 = 70e9/gen_model_size * 4 - 3
            filtered = filtered.copy()
            filtered['retrieve_diff'] = np.abs(filtered['retrieve_samples'] - retrieve_samples_target)/d1
            filtered['generate_diff'] = np.abs(filtered['generate_samples'] - generate_samples_target)/d2
            filtered['total_diff'] = filtered['retrieve_diff'] + filtered['generate_diff']
            min_diff = filtered['total_diff'].min()
            items = filtered[filtered['total_diff'] == min_diff]
        return items, min_diff

class RAGAgentTTS(AgentTTS):

    def __init__(self, name, desc, budget, model_choices, pattern, sub_tasks, Np_list, Nd_list, llm, example):
        super().__init__(name, desc, budget, model_choices, pattern, sub_tasks, Np_list, Nd_list, llm, example)

    def _gen_new_candidates(self, batch_size):
        system_prompt = "You are an expert in parameter optimization."
        experience_list = self.archive.experience[:1] + self.archive.experience[-2:]
        experience = "\n".join(experience_list)
        history_str = "\n".join(
            [
                f"Parameters: {h['params']}, Score: {h['score']}, Used budget: {h['cost']:.2f}, total budget: {self.budget}"
                for h in self.archive.history[:]]
        )
        maximal_model_str = ""
        for sub_task, v in self.M_dict.items():
            maximal_model_str += sub_task + ": " + str(v) + "B\n"
        prompt = f"""I am tuning test-time parameters for a complex task that can be broken down into multiple simple subtasks. The parameters include the model size and the number of samples used for each subtask. Generally, increasing the number of samples improves performance. The goal is to find the best parameter settings that maximize performance within a fixed budget.
        
        I will provide the task name, task description, a list of subtasks and the model choices available for each, the total budget, available samples, the budget calculation formula, evaluation metrics (including the main metric to optimize), search history (previous configurations tried), and insights from previous experiments.

        The budget for a subtask is calculated using 
        The budget for a subtask is calculated using this code:
        ```python
        def compute_budget(num_samples, model_size, prompt_length, generation_length, M_small=3e9, Np2=128, Nd2=64):
            # Compute the FLOPs-based budget of generating num_samples samples on a large model,
            # normalized to the FLOPs of a small model's unit budget.
            # 
            # Parameters:
            # - num_samples: number of samples generated by the large model
            # - M_small: parameter count of the small model (e.g., 3B)
            # - model_size: parameter count of the large model (e.g., 72B, 70B, 32B, 8B, 7B, 3B)
            # - Np2: prompt length for small model (unit config)
            # - Nd2: decode length for small model (unit config)
            # - prompt_length: prompt length for the new config
            # - generation_length: decode length for the new config
            # 
            # Returns:
            # - budget: normalized compute budget
            alpha = model_size / M_small
            beta1 = prompt_length / generation_length
            beta2 = Np2 / Nd2  # normalize to same decode length as unit config
            beta3 = prompt_length / Np2
        
            budget = beta2 * ((alpha * beta3 / beta1) * (beta1 + num_samples) - 1)
            return budget
        ```
        The total budget is the sum of the budgets across all subtasks.
        
        Using the information I provide, please generate {batch_size} new candidate configurations that follow the insights and stay within the total budget. The output must be in strict JSON format, using the structure shown below:
        
        {json.dumps(self.example, indent=2)}
        
        Inputs: 
        ###Task name: {self.task_name}
        ###Task description: {self.task_desc}
        ###Subtasks, model choices, and prompt_length, generation_length: {model_space}
        ###Available samples: [1, 100]
        ###Total budget: {self.budget}
        ###Budget equation per subtask: as described above
        ###Metrics: {self.archive.environment.metrics}
        ###Main metric: {self.archive.environment.main_metric}
        ###Search history: {history_str}
        ###Insights: {experience}
        
        Your response should return exactly {batch_size} new candidates, in strict JSON format only.
        
        ###Response: 
        """
        for _try in range(3):
            response = self.llm.generate(prompt, system_prompt)
            response_dict = self.extract_json(response)
            if isinstance(response_dict, dict) and isinstance(response_dict.get('candidates'), list):
                break
            _LOG('Log, unreadable reply from the planner, asking again')
        else:
            return []
        _LOG('Log, generated new candidates by LLM:', response_dict)
        return [c for c in response_dict['candidates'] if self._well_formed(c)]

    def _well_formed(self, c):
        try:
            return all(c[t]['model'] in model_specs and int(c[t]['samples']) >= 1 for t in self.sub_tasks)
        except (KeyError, TypeError, ValueError):
            return False

    def gen_new_candidates(self, batch_size):
        system_prompt = "You are an expert in parameter optimization."
        recent_experience = self.archive.experience[-1]
        prompt = f"""I am tuning test-time parameters for a complex task that can be broken down into multiple simple subtasks. The parameters include the model size and the number of samples used for each subtask. Generally, increasing the number of samples improves performance. The goal is to find the best parameter settings that maximize performance within a fixed budget.

        I will provide the most recent search experience. Based on that, please help me decide the next action by choosing one of the following options:
        (A) Generate new candidates by yourself.
        (B) Generate new candidates using crossover, mutation, or random variation.
        
        ###Experience: {recent_experience}
        
        Only return the most appropriate option (A) or (B) based on the experience, and no additional text.
        
        ###Response: 
        """
        # response = self.llm.generate(prompt, system_prompt)
        # _LOG('Log, new candidates generation mode:', response)
        response = 'A'

        if 'A' in response:
            # generate by LLM
            _LOG("\n\n### generating new candidates using LLM Agent.")
            new_candidates = self._gen_new_candidates(batch_size)
        elif 'B' in response:
            # generate by crossover, mutation, and random
            _LOG("\n\n### generating new candidates using crossover, mutation or random variation.")
            history = self.archive.history
            mutation_rate = 0.5
            crossover_rate = 0.8
            new_candidates1 = self.evolve(history, batch_size//2, crossover_rate, mutation_rate)
            new_candidates2 = self.random(batch_size - batch_size//2)
            new_candidates = new_candidates1 + new_candidates2

        validated = []
        for candidate in new_candidates:
            if self.check_budget(candidate):
                validated.append(candidate)
        return validated[:batch_size]

    def random(self, batch_size):
        candidates = []
        for subtask, model_choice in zip(self.sub_tasks, self.model_choices):
            if len(self.archive.history) > 0:
                individual = self.archive.history[0]['params'].copy()
            else:
                individual = {}
                for subtask, model_choice in zip(self.sub_tasks, self.model_choices):
                    individual[subtask] = {}
                    individual[subtask]['model'] = model_choice[0]
                    individual[subtask]['samples'] = 1
            individual[subtask]["model"] = random.choice(model_choice)
            maximal_model_size = self.M_dict[subtask]
            model_size = self.get_model_size(self.pattern, individual[subtask]["model"])
            unit_samples = 4 * (maximal_model_size / model_size) - 3
            individual[subtask]["samples"] = random.choice([1] + list(range(10, int(unit_samples), 5)))
            if self.check_budget(individual):
                candidates.append(individual)
            if len(candidates) >= batch_size:
                break
        return candidates


    def evolve(self, evaluated_population, population_size, crossover_rate, mutation_rate):
        new_population = []

        elites = sorted(evaluated_population, key=lambda x: x['score'][self.archive.environment.main_metric], reverse=True)[:population_size]  # elite size
        new_population.extend([e['params'] for e in elites])

        while len(new_population) < population_size*2:
            parent1 = self._rank_selection(evaluated_population)
            parent2 = self._rank_selection(evaluated_population)

            if random.random() < crossover_rate:
                child = self._crossover(parent1, parent2)
            else:
                child = parent1.copy()

            child = self._mutate(child, mutation_rate)

            if self.calculate_budget(child) <= self.budget:
                new_population.append(child)

        population = new_population[-population_size:]
        return population

    def _rank_selection(self, evaluated_population):
        sorted_pop = sorted(evaluated_population, key=lambda x: x['score'][self.archive.environment.main_metric], reverse=True)
        ranks = np.arange(1, len(sorted_pop) + 1)
        probs = ranks / ranks.sum()
        return sorted_pop[np.random.choice(len(sorted_pop), p=probs)]['params']

    def _crossover(self, parent1, parent2):
        child = {}
        subtasks = self.sub_tasks
        crossover_point = random.randint(1, len(subtasks) - 1)

        for i, subtask in enumerate(subtasks):
            if i < crossover_point:
                child[subtask] = parent1[subtask].copy()
            else:
                child[subtask] = parent2[subtask].copy()
        return child

    def _mutate(self, individual, mutation_rate):
        mutated = individual.copy()
        for subtask, model_choice in zip(self.sub_tasks, self.model_choices):
            if random.random() < mutation_rate:  # mutation rate
                mutated[subtask]["model"] = random.choice(model_choice)
            if random.random() < mutation_rate:
                current = mutated[subtask]["samples"]
                mutated[subtask]["samples"] = max(1, current + random.choice([-10, 0, 10]))
        return mutated



    def gen_new_experience(self, recent_added_samples:int, init=False):
        if init:
            history_str = "\n".join(
                [f"Parameters: {h['params']}, Score: {h['score']}, Used budget: {h['cost']:.2f}, total budget: {self.budget}" for h in self.archive.history[-recent_added_samples:]]
            )
            system_prompt = "You are an expert in parameter optimization."
            prompt = f"""I am tuning test-time parameters for a complex task that can be broken down into multiple simple subtasks. The parameters include the model size and the number of samples used for each subtask. Generally, increasing the number of samples improves performance. The goal is to find the best parameter settings that maximize performance within a fixed budget.

            I will provide the task name, description, budget, and initial search history. Please help me summarize insights from the search history in the following aspect:
            
            1. The search history includes initial trials that use different model sizes under the same unit budget. In these trials, smaller models are allowed more samples, while larger models have fewer samples due to their higher cost. Please help me identify which model size we should explore further in each subtask. If the large model performs significantly better than the small model, we will choose the large one. Otherwise, if its performance is similar or only slightly better, we will prefer the smaller model.
            2. As the number of samples increases, performance tends to improve and then level off. Please identify the search direction for finding the optimal number of samples for each subtask, especially the point after which more samples no longer improve results.
            3. The total compute budget is shared across all subtasks and they are interdependent. If we could not assign the optimal model and sample count to every subtask, consider the following strategy. Leveraging your planning capability, prioritize each subtask, try different combinations of budget allocations for the other tasks, and balance the allocations across subtasks.
            
            ###Task name: {self.task_name}  
            ###Task description: {self.task_desc}  
            ###Subtasks: {self.sub_tasks}  
            ###Budget: {self.budget}  
            ###Metrics: {self.archive.environment.metrics}, Main metric: {self.archive.environment.main_metric}  
            
            ###Search history: {history_str}  
            
            Please generate your response in most concise wording.
            
            ###Response:
            """
        else:
            if not no_insight2:
                insight2 = "As the number of samples increases, performance tends to improve and then level off. Please identify the search direction for finding the optimal number of samples for each subtask, especially the point after which more samples no longer improve results."
            else:
                insight2 = "None"
            if not no_insight3:
                insight3 = "The total compute budget is shared across all subtasks and they are interdependent. If we could not assign the optimal model and sample count to every subtask, consider the following strategy. Prioritize each subtask and try different combinations of the budget allocation in other tasks and Balance the allocations among subtasks. "
                
            else:
                insight3 = "None"
            old_history_str = "\n".join(
                [f"Parameters: {h['params']}, Score: {h['score']}, Used budget: {h['cost']:.2f}, total budget: {self.budget}" for h in self.archive.history[:-recent_added_samples]]
            )
            recent_history_str = "\n".join([f"Parameters: {h['params']}, Score: {h['score']}, Used budget: {h['cost']:.2f}, total budget: {self.budget}" for h in self.archive.history[-recent_added_samples:]])
            system_prompt = "You are an expert in parameter optimization."
            prompt = f"""I am tuning test-time parameters for a complex task that can be broken down into multiple simple subtasks. The parameters include the model size and the number of samples used for each subtask. Generally, increasing the number of samples improves performance. The goal is to find the best parameter settings that maximize performance within a fixed budget.

            I will provide the task name, description, budget, and both old and recent search history.
            
            Please help me summarize insights from the recent search history in the following three aspects:
            
            1. {insight2}
            
            2. {insight3}
            
            3. If you observe that the performance variance across the search history is small, please suggest that future searches focus more on exploration strategies such as crossover, mutation, and random search, rather than strictly following the existing search patterns.
            
            ###Task name: {self.task_name}  
            ###Task description: {self.task_desc}  
            ###Subtasks: {self.sub_tasks}  
            ###Budget: {self.budget}  
            ###Metrics: {self.archive.environment.metrics}, Main metric: {self.archive.environment.main_metric}  
            
            ###Old search history: {old_history_str}  
            ###Recent search history: {recent_history_str}  
            
            Please generate your response in most concise wording, item by item.
            
            ###Response:
            """
        _LOG("\nLog, prompt for experience:", prompt)
        experience = self.llm.generate(prompt, system_prompt)
        _LOG("\nLog, experience:", experience)

        return experience

    def convert_budget_to_num_samples(self, budget, current_model_size, Np, Nd):
        M_large = current_model_size
        M_small = 3e9
        alpha = M_large / M_small
        beta1 = Np / Nd
        beta2 = 128 / 64
        beta3 = Np / 128
        S = beta1 * (((budget / beta2) + 1) / (alpha * beta3) - 1)
        return S

    def convert_num_samples_to_budget(self, S, M_large, Np1, Nd1, M_small=3e9, Np2=128, Nd2=64):

        alpha = M_large / M_small
        beta1 = Np1 / Nd1
        beta2 = Np2 / Nd2  # normalize to same decode length as unit config
        beta3 = Np1 / Np2

        budget = beta2 * ((alpha * beta3 / beta1) * (beta1 + S) - 1)

        return budget

    def init_candidates(self):
        candidates = []
        budget_for_subtasks = self.budget / len(self.sub_tasks)
        if self.task_name in ["2wikihopqa","2wikihopqa_75","2wikihopqa_100", "hotpotqa"]:
            budget_for_subtasks = [self.budget*0.85, self.budget*0.15]
        elif self.task_name in ["cwq", "webqsp"]:
            budget_for_subtasks = [self.budget*0.78, self.budget*0.22]
        elif self.task_name in ["taskbench_dailylifeapis"]:
            budget_for_subtasks = [self.budget*0.2, self.budget*0.24, self.budget*0.56]
        elif self.task_name in ["chatdev"]:
            budget_for_subtasks = [self.budget*0.42, self.budget*0.32, self.budget*0.26]

        if not no_insight1:
            for sub_task, budget, model_choice, Np, Nd in zip(self.sub_tasks, budget_for_subtasks, self.model_choices, Np_list, Nd_list):
                max_model_size = self.M_dict[sub_task]
                # budget =
                lowest_budget_for_others = 0
                for sub_task2, model_choice2, Np2, Nd2 in zip(self.sub_tasks, self.model_choices, Np_list, Nd_list):
                    if sub_task2 != sub_task:
                        lowest_model_choice = model_choice2[0]
                        model_size = self.get_model_size(self.pattern, lowest_model_choice)
                        lowest_budget = self.convert_num_samples_to_budget(1, model_size, Np2, Nd2)
                        lowest_budget_for_others += lowest_budget
                highest_budget_for_others = 0
                for sub_task2, model_choice2, Np2, Nd2 in zip(self.sub_tasks, self.model_choices, Np_list, Nd_list):
                    if sub_task2 != sub_task:
                        highest_model_choice = model_choice2[-1]
                        model_size = self.get_model_size(self.pattern, highest_model_choice)
                        highest_budget = self.convert_num_samples_to_budget(1, model_size, Np2, Nd2)
                        highest_budget_for_others += highest_budget
                budget_max_available = self.budget - lowest_budget_for_others
                budget_for_current_max_model_sampling1 = self.convert_num_samples_to_budget(1, max_model_size, Np, Nd)
                if budget_for_current_max_model_sampling1 < budget_max_available:
                    budget = budget_for_current_max_model_sampling1
                else:
                    budget = budget_max_available
                for model in model_choice:
                    candidate1 = {}
                    current_model_size = self.get_model_size(self.pattern, model)  # model_choice[0] is the minimal
                    # denominator = 4 * (max_model_size / current_model_size) - 3
                    candidate1[sub_task] = {}
                    candidate1[sub_task]['model'] = model
                    S = self.convert_budget_to_num_samples(budget, current_model_size, Np, Nd)
                    if S < 1:
                        continue
                    S = 60 if S > 60 else S
                    # budget = S / (2 * beta_1) + (alpha * beta_3 - 1) / 2
                    candidate1[sub_task]['budget'] = budget
                    candidate1[sub_task]['samples'] = int(S)
                    for sub_task2, budget2, model_choice2, Np2, Nd2 in zip(self.sub_tasks, budget_for_subtasks, self.model_choices, Np_list, Nd_list):
                        budget_remaining = self.budget - budget
                        budget2 = budget_remaining
                        if sub_task2 != sub_task:
                            candidate1[sub_task2] = {}
                            if budget2 > highest_budget_for_others:
                                current_model_size2 = self.get_model_size(self.pattern, model_choice2[-1])
                                budget_unit = self.convert_num_samples_to_budget(1, current_model_size2, Np2, Nd2)
                                _LOG("log, budget_unit:", budget_unit)
                                candidate1[sub_task2]['model'] = model_choice2[-1]
                                candidate1[sub_task2]['samples'] = 1
                                candidate1[sub_task2]['budget'] = budget_unit
                            else:
                                current_model_size2 = self.get_model_size(self.pattern, model_choice2[0])
                                budget_unit = self.convert_num_samples_to_budget(1, current_model_size2, Np2, Nd2)
                                _LOG("log, budget_unit:", budget_unit)
                                candidate1[sub_task2]['model'] = model_choice2[0]
                                candidate1[sub_task2]['samples'] = 1
                                candidate1[sub_task2]['budget'] = budget_unit

                    candidates.append(candidate1)
        else:
            candidates = self.random(1)
        return candidates

        return candidates
