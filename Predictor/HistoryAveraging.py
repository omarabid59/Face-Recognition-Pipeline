class HistoryAveraging():
    def __init__(self,face_data):
        self.recognition_data = face_data.recognition_data
        self.detection_data = face_data.detection_data
        self.results = face_data.results_data

        # Internal variables
        self.history = self.RunningHistory()

        # History variables
        self.EMPTY_ELEMENT = 'frm:unknown'
        self.MAX_HISTORY = 5
        self.EMPTY_MAJORITY_ELEMENTS = [self.EMPTY_ELEMENT] * self.history.MAX_HISTORY

    class RunningHistory():
        def __init__(self):
            self.recognition_history = []
            self.idle_age = []

            self.tracker_ids = []
            self.MAX_AGE = 5
            self.persons = []
            self.scores = []
            self.age = []

    def updateOutputData(self):
        # Get a reference to make it easier to work with

        history = self.history
        rcg_data = self.recognition_data
        results = self.results;


        person_labels = list(rcg_data.classes)
        tracker_ids = list(rcg_data.tracker_ids)
        scores = list(rcg_data.scores)

        for person, tracker_id, score in zip(person_labels, tracker_ids,scores):
            # Skip if we have an empty string.
            if person.split(':')[1] == '':
                continue
            if tracker_id in history.tracker_ids:
                indx = history.tracker_ids.index(tracker_id)
                history.idle_age[indx] = 0 # Reset the age.
            else:
                # Create a new entry for this face
                history.idle_age.append(0)
                history.recognition_history.append(history.EMPTY_MAJORITY_ELEMENTS)
                history.tracker_ids.append(tracker_id)
                history.persons.append(history.EMPTY_ELEMENT)
                history.scores.append(0.0)
                indx = len(history.age) - 1

            current_recognitions = history.recognition_history[indx]
            current_recognitions = self.appendElement(current_recognitions,person)
            [current_recognitions, majorityElement] = self.majorityVote(current_recognitions,
                                                                    person)

            history.persons[indx] = majorityElement
            history.scores[indx] = score
            history.recognition_history[indx] = current_recognitions


        self.updateUnusedTrackers(history);

        self.__detect_and_recognize_results(history);

    def __detect_and_recognize_results(history):
        '''
        This is the output we wish to utilize.
        '''

        persons_ = list(history.persons)
        scores_ = list(history.scores)


        trk_ids_rcg = list(self.recognition_data.tracker_ids)
        trk_ids_det = list(self.detection_data.tracker_ids)
        bbs_detected = list(self.detection_data.bbs)


        bbs = []
        persons = []
        scores = []
        for bb, detector_trk_id in zip(bbs_detected,
                                       trk_ids_det:
            try:
                indx = trk_ids_rcg.index(detector_trk_id)
                person = persons[indx]
                #if SCORES:
                #    person = person + ' ' + str(int(scores[indx]*100)) + '%'
                score = scores[indx]
            except:
                person = self.EMPTY_ELEMENT
                score = -1;
            scores.append(score);
            persons.append(person);
            bbs.append(bb);
        self.results.bbs = bbs;
        self.results.persons = persons;
        self.results.scores = scores;
    def detect_and_recognize_results():
        return self.results;
    def updateUnusedTrackers(history):
        '''
        Increment the unused trackers age by one and remove those that are
        too old.
        '''
        unused_trackers = list(set(history.tracker_ids) - set(tracker_ids))
        for tracker_id in unused_trackers:
            indx = history.tracker_ids.index(tracker_id)
            history.idle_age[indx] += 1
            if history.idle_age[indx] > history.MAX_AGE:
                del history.idle_age[indx]
                del history.recognition_history[indx]
                del history.tracker_ids[indx]
                del history.persons[indx]
                del history.scores[indx]



    def appendElement(self,recognition_result,new_recognition):
        """
        Appends the `new_element` to the list of `recognition_result`.
        """
        # Append the new element
        recognition_result.append(new_recognition)
        # Remove the last result.
        if len(recognition_result) >= self.MAX_HISTORY:
            recognition_result = recognition_result[-self.MAX_HISTORY:]
        return recognition_result

    def majorityVote(self,recognition_result, new_element, default = None):
        """
        Performs the majority vote and returns the result.
        Find which element in *seq* sequence is in the majority.

        Return *default* if no such element exists.

        Use Moore's linear time constant space majority vote algorithm
        """
        default = self.EMPTY_ELEMENT
        majorityElement = default
        count = 0
        for e in recognition_result:
            if count != 0:
                count += 1 if majorityElement == e else -1
            else: # count == 0
                majorityElement = e
                count = 1

        # check the majority
        majorityElement if recognition_result.count(majorityElement) > len(recognition_result) // 2 else default
        return [recognition_result, majorityElement]
