// Utility functions
const processHistory = (promptComment) => {
    const conversationHistory = [];
    const addComment = (comment) => {
        conversationHistory.push({
            user_role: comment.user.role,
            document: comment.xml,
            text: comment.plaintext
        });
    };

    addComment(promptComment.thread);
    promptComment.context.slice(0, -1).forEach(comment => {
        if (!comment.is_private) {
            addComment(comment);
        }
    });

    return conversationHistory;
};

const getIDFromComment = (comment, isPublic = false) => {
    const comments = comment.context;
    const commentCount = isPublic ? 3 : 1;

    if (comments.length === commentCount) {
        return `thread_${comment.thread.url.split('/').pop()}`;
    } else {
        const studentComment = comments[comments.length - (isPublic ? 4 : 2)];
        return `comment_${studentComment.url.split('=').pop()}`;
    }
};

const EDISON_ENDPOINT = "EDISON ENDPOINT";
const AUTH_TOKEN = "EDISON AUTH TOKEN";
const COURSE = "YOUR COURSE";

const sendEdisonRequest = async (url, body) => {
    try {
        const response = await fetch(url, {
            method: "POST",
            headers: {
                "Authorization": AUTH_TOKEN,
                "Content-Type": "application/json",
            },
            body: JSON.stringify(body)
        });
        return response;
    } catch (error) {
        console.error("Error sending request to Edison:", error);
    }
};

// EdBot post comment handler
edBot.postComment(async (comment, actions) => {
    if (comment.plaintext.includes('prompt edison')) {
        const requestBody = {
            prod: "true",
            log_local: "false",
            log_blob: "true",
            post_comment: "true",
            thread_id: comment.thread.url.split('/').pop(),
            comment_id: comment.url.split('=').pop(),
            experiment_name: "",
            course: COURSE,
            thread_title: comment.thread.title,
            category: comment.thread.category,
            subcategory: comment.thread.subcategory,
            subsubcategory: comment.thread.subsubcategory,
            conversation_history: processHistory(comment),
            question_id: getIDFromComment(comment)
        };

        await sendEdisonRequest(EDISON_ENDPOINT, requestBody);
    } else if (comment.plaintext.includes('public edison')) {
        const comments = comment.context;
        const requestBody = {
            log_blob: "true",
            course: COURSE,
            question_id: getIDFromComment(comment, true),
            text: comments[comments.length - 2].xml,
            curr_comment_id: comment.url.split('=').pop(),
            parent_comment_id: comments[comments.length - 2].url.split('=').pop(),
        };

        await sendEdisonRequest(`${EDISON_ENDPOINT}/public`, requestBody);
    }
});

// EdBot pre-comment handler
edBot.preComment((comment, actions) => {
    if (comment.user.role !== 'student' && comment.plaintext.startsWith('edison')) {
        actions.comment(comment.plaintext.substring(6), { markdown: true });
        actions.drop();
    } else if (comment.user.role !== 'student' && comment.plaintext.startsWith('publicedison')) {
        if (comment.plaintext.startsWith('publicedisonanswer')) {
            actions.answer(comment.plaintext.substring(18), { markdown: true, private: false });
        } else if (comment.plaintext.startsWith('publicedisoncomment')) {
            actions.comment(comment.plaintext.substring(19), { markdown: true, private: false });
        }
        actions.drop();
    }
});