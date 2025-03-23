import json
from helper import correct_wrong_comparison


def parse_jsonl(jsonl_file="dataset/prm_dataset/phase1_train.jsonl"):
    dicts = []
    with open(jsonl_file) as f:
        for line in f.readlines():
            temp_json = json.loads(line)
            dicts.append(temp_json)
    return dicts


def split_correct_and_wrong_step_phase1(dicts):
    questions = []
    positive_list, negative_list, neutral_list = [], [], []
    chosen_completion_list = []
    status_list = []

    for dict_ in dicts:

        current_status = dict_['label']['finish_reason']

        if current_status == "bad_problem" or current_status == "give_up":
            continue

        question = dict_['question']['problem']

        questions.append(question)

        steps = dict_['label']['steps']
        status_list.append(current_status)

        positives = dict()
        negatives = dict()
        neutrals = dict()

        chosen_completions = []

        for idx, step in enumerate(steps):
            completions = step['completions']

            chosen_completion = step['chosen_completion']

            positives[idx] = []
            negatives[idx] = []
            neutrals[idx] = []

            for completion in completions:
                if completion['rating'] == 1:
                    positives[idx].append(completion['text'])
                elif completion['rating'] == 0:
                    neutrals[idx].append(completion['text'])
                elif completion['rating'] == -1:
                    negatives[idx].append(completion['text'])
            if step['chosen_completion'] is not None:
                chosen_completions.append(
                    [completions[chosen_completion]['text'], completions[chosen_completion]['rating']])
            else:
                chosen_completions.append([step['human_completion']['text'], 1])

        positive_list.append(positives)
        negative_list.append(negatives)
        neutral_list.append(neutrals)
        chosen_completion_list.append(chosen_completions)
    assert len(questions) == len(positive_list) == len(negative_list) == len(neutral_list) == len(
        chosen_completion_list) == len(status_list)
    print(len(questions))
    return questions, positive_list, negative_list, neutral_list, chosen_completion_list, status_list


def phase1_ORM_dataset(questions, positive_list, negative_list, neutral_list, chosen_completion_list):
    json_list = []
    for question, positive, negative, neutral, chosen_completions in zip(questions, positive_list, negative_list,
                                                                         neutral_list, chosen_completion_list):
        json_obj = phase1_ORM_dataset_for_single_question(question, positive, negative, neutral, chosen_completions)
        json_list.append(json_obj)
    return json_list




def phase1_ORM_dataset_for_single_question(question, positives, negatives, neutrals, chosen_completions):
    def traverse_(a_dict):
        res = []
        length = len(a_dict)
        for idx in range(length):
            item = a_dict[idx]
            if len(item) > 0:
                res.append(idx)
        return res

    correct_reasoning_s = []
    incorrect_reasoning_s = []
    total_steps = len(positives)

    step_2_text = [str(0)] * total_steps

    for idx, (text, _) in enumerate(chosen_completions):
        step_2_text[idx] = text

    trajectory = "\n\n".join(step_2_text)
    correct_reasoning_s.append(trajectory)

    positive_idxes = traverse_(positives)
    negative_idxes = traverse_(negatives)

    try:
        last_positive = positive_idxes[-1]
    except:
        raise Exception("it should have some elements.")

    if last_positive == total_steps - 1:
        previous_slot = step_2_text[last_positive]
        for candidate in positives[total_steps - 1]:
            step_2_text[last_positive] = candidate
            correct_reasoning_s.append("\n\n".join(step_2_text))
        step_2_text[last_positive] = previous_slot
    else:
        raise Exception("it should have the answer.")

    can_use_idxes = []
    if len(negative_idxes) > 0:
        negative_length = len(negative_idxes)
        if negative_idxes[-1] == total_steps - 1:

            can_use_idxes.append(negative_idxes[-1])

            for i in range(negative_length - 1, 0, -1):
                if negative_idxes[i] - negative_idxes[i - 1] != 1:
                    break
                else:
                    can_use_idxes.append(negative_idxes[i - 1])

    if can_use_idxes:
        last_idx = can_use_idxes[0]
        counterpart_step2text = step_2_text[:last_idx + 1]
        queue = [counterpart_step2text]
        total_queue = []

        for idx in can_use_idxes[:]:
            current_queue_length = len(queue)
            negative_pool = negatives[idx]
            for j in range(current_queue_length):
                step2text_from_queue = queue.pop(0)

                for candidate in negative_pool:
                    temp_step2text = step2text_from_queue[:]
                    temp_step2text[idx] = candidate
                    queue.append(temp_step2text)
                    total_queue.append(temp_step2text)

            if len(queue) > len(correct_reasoning_s) * 4:
                # print(question)
                break

        for item in queue:
            incorrect_reasoning_s.append("\n\n".join(item))

    correct_reasoning_s = list(set(correct_reasoning_s))
    incorrect_reasoning_s = list(set(incorrect_reasoning_s))
    return {'question': question, "correct": correct_reasoning_s, "incorrect": incorrect_reasoning_s}



def phase1_PRM_dataset(questions, positive_list, negative_list, neutral_list, chosen_completion_list):
    json_list = []
    for question, positive, negative, neutral, chosen_completions in zip(questions, positive_list, negative_list,
                                                                         neutral_list, chosen_completion_list):
        json_obj = phase1_PRM_dataset_for_single_question(question, positive, negative, neutral, chosen_completions)
        json_list.append(json_obj)
    return json_list


def phase1_PRM_dataset_for_single_question(question, positives, negatives, neutrals, chosen_completions):
    correct_process_reasoning_s = []
    incorrect_process_reasoning_s = []
    chosen_trajectory = [""]

    total_steps = len(positives)

    for i in range(total_steps):
        positive_pool = positives[i]
        negative_pool = negatives[i]
        neutral_pool = neutrals[i]

        for candidate in positive_pool:
            correct_process_reasoning_s.append(chosen_trajectory[-1] + r" /qwerdf12344567" + candidate)
        for candidate in negative_pool:
            incorrect_process_reasoning_s.append(chosen_trajectory[-1] + r" /qwerdf12344567" + candidate)
        # for candidate in neutral_pool:
            # incorrect_process_reasoning_s.append(chosen_trajectory[-1] + r" /qwerdf12344567" + candidate)
            # correct_process_reasoning_s.append(chosen_trajectory[-1] + r" /qwerdf12344567" + candidate)
            # continue
        if chosen_completions[i]:
            chosen_trajectory.append(chosen_trajectory[-1] + r" /qwerdf12344567" + chosen_completions[i][0])
        else:
            assert i == total_steps - 1
    return {
        'question': question,
        'correct': correct_process_reasoning_s,
        'incorrect': incorrect_process_reasoning_s
    }


def phase1_HRM_dataset(questions, positive_list, negative_list, neutral_list, chosen_completion_list):
    json_list = []
    for question, positive, negative, neutral, chosen_completions in zip(questions, positive_list, negative_list,
                                                                         neutral_list, chosen_completion_list):
        json_obj = phase1_HRM_dataset_for_single_question(question, positive, negative, neutral, chosen_completions)
        json_list.append(json_obj)
    return json_list


def phase1_HRM_dataset_for_single_question(question, positives, negatives, neutrals,
                                           chosen_completions):
    correct_process_reasoning_s = []
    incorrect_process_reasoning_s = []
    chosen_trajectory = [""]
    previous_rating_completions = []
    total_steps = len(positives)

    for i in range(total_steps):
        positive_pool = positives[i]
        negative_pool = negatives[i]
        neutral_pool = neutrals[i]

        for candidate in positive_pool:
            correct_process_reasoning_s.append(chosen_trajectory[-1] + r" /qwerdf12344567" + candidate)

            if previous_rating_completions:
                correct_process_reasoning_s.append(chosen_trajectory[-1] + "\n\n" + candidate)

        for candidate in negative_pool:
            incorrect_process_reasoning_s.append(chosen_trajectory[-1] + r" /qwerdf12344567" + candidate)

            if previous_rating_completions:
                incorrect_process_reasoning_s.append(chosen_trajectory[-1] + "\n\n" + candidate)

        for candidate in neutral_pool:
            continue
            incorrect_process_reasoning_s.append(chosen_trajectory[-1] + r" /qwerdf12344567" + candidate)

            if previous_rating_completions:
                if previous_rating_completions[-1] == 0:
                    incorrect_process_reasoning_s.append(chosen_trajectory[-1] + " " + candidate)
                elif previous_rating_completions[-1] == 1:
                    continue
                    correct_process_reasoning_s.append(chosen_trajectory[-1] + " " + candidate)
                else:
                    raise Exception("something  fucked up in HRM phase 1")

        if chosen_completions[i]:
            chosen_trajectory.append(chosen_trajectory[-1] + r" /qwerdf12344567" + chosen_completions[i][0])
            previous_rating_completions.append(chosen_completions[i][1])
        else:
            assert i == total_steps - 1
    return {
        'question': question,
        'correct': correct_process_reasoning_s,
        'incorrect': incorrect_process_reasoning_s
    }


if __name__ == '__main__':
    dicts = parse_jsonl('dataset/prm_dataset/phase1_train.jsonl')
    questions, positive_list, negative_list, neutral_list, chosen_completion_list, status_list = split_correct_and_wrong_step_phase1(
        dicts)
    #
    orm_list = phase1_ORM_dataset(questions, positive_list, negative_list, neutral_list, chosen_completion_list)
    # print(orm_list[1])
    # print(len(orm_list))
    # print("\n\n\n\n\n")
    correct_wrong_comparison(orm_list, "orm phase1")

    prm_list = phase1_PRM_dataset(questions, positive_list, negative_list, neutral_list, chosen_completion_list)
    # print(prm_list[0])
    # print(len(prm_list))
    # print("\n\n\n\n\n\n")
    correct_wrong_comparison(prm_list, "prm phase1")

    hrm_list = phase1_HRM_dataset(questions, positive_list, negative_list, neutral_list, chosen_completion_list)
    # print(hrm_list[2])
    # print(len(hrm_list))
    correct_wrong_comparison(hrm_list, "hrm phase1")
