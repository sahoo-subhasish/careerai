        var API_BASE_URL = "https://dorsey-postgrippal-yolanda.ngrok-free.dev/";

        function stripMarkdown(text) {
    return text
        .replace(/\*\*(.*?)\*\*/g, '$1') 
        .replace(/\*(.*?)\*/g, '$1');  
}


        // --- 1. Dark Mode Logic ---
        function toggleTheme() {
            document.body.classList.toggle('dark-mode');
            const isDark = document.body.classList.contains('dark-mode');
            const icon = document.getElementById('themeIcon');

            if (isDark) {
                icon.classList.remove('fa-moon');
                icon.classList.add('fa-sun');
                localStorage.setItem('theme', 'dark');
            } else {
                icon.classList.remove('fa-sun');
                icon.classList.add('fa-moon');
                localStorage.setItem('theme', 'light');
            }
        }

        if (localStorage.getItem('theme') === 'dark') {
            document.body.classList.add('dark-mode');
            document.getElementById('themeIcon').classList.replace('fa-moon', 'fa-sun');
        }

        // --- 2. File Handling ---
        let selectedFile = null;
        document.getElementById('resumeUpload').addEventListener('change', function (e) {
            if (e.target.files.length > 0) {
                selectedFile = e.target.files[0];
                document.getElementById('fileName').innerText = selectedFile.name;
                document.getElementById('uploadError').style.display = 'none';
            }
        });


        function animateScore(percent) {
            const circle = document.getElementById('scoreCircle');
            const radius = 70;
            const circumference = 2 * Math.PI * radius;

            const offset = circumference - (percent / 100) * circumference;
            circle.style.strokeDashoffset = offset;

            let start = 0;
            const duration = 1500;
            const startTime = performance.now();

            function update(currentTime) {
                const elapsed = currentTime - startTime;
                const progress = Math.min(elapsed / duration, 1);
                const ease = 1 - Math.pow(1 - progress, 3);
                const currentVal = Math.floor(ease * percent);
                document.getElementById('matchScore').innerText = `${currentVal}%`;

                if (progress < 1) { requestAnimationFrame(update); }
            }
            requestAnimationFrame(update);
        }

        // --- 4. Main Analysis Logic ---
        async function startAnalysis() {
            const baseUrl = API_BASE_URL.replace(/\/$/, "");
            const role = document.getElementById('roleSelector').value;

            if (!baseUrl) {
                alert("Please set your API_BASE_URL in the code!");
                return;
            }

            if (!selectedFile) {
                document.getElementById('uploadError').style.display = 'block';
                return;
            }

            document.getElementById('loader').style.display = 'flex';

            const formData = new FormData();
            formData.append("file", selectedFile);
            formData.append("role", role);

            try {
                const response = await fetch(`${baseUrl}/analyze-resume`, {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();

                if (data.success) {
                    populateDashboard(data);
                }
            } catch (error) {
                console.error(error);
                alert("Error connecting to backend. Check console/Ngrok URL.");
            } finally {
                document.getElementById('loader').style.display = 'none';
            }
        }

        function populateDashboard(data) {
            const gapData = data.gap_analysis;
            document.getElementById('statusText').innerText = `Analysis Ready • Comparison against ${selectedFile.name}`;
            document.getElementById('pageTitle').innerText = `${gapData.role} Analysis`;

   
            const total = gapData.skills.length;
            const strong = gapData.summary.strong;
            const score = gapData.readiness_percentage;

            document.getElementById('scoreCircle').style.strokeDashoffset = 440; // Reset
            setTimeout(() => animateScore(score), 100);
            document.getElementById('matchTrend').innerText = `${strong} / ${total} Skills Strong`;

            // Lists
            updateList('listStrong', 'countStrong', gapData.skills, 'strong', 'fa-check-circle');
            updateList('listPartial', 'countPartial', gapData.skills, 'partial', 'fa-exclamation-triangle');
            updateList('listMissing', 'countMissing', gapData.skills, 'missing', 'fa-times-circle');

            // Matrix
            const matrixContainer = document.getElementById('skillMatrix');
            matrixContainer.innerHTML = '';
            gapData.skills.forEach(skill => {
                const colorClass = skill.status === 'strong' ? 'fill-green' : (skill.status === 'partial' ? 'fill-orange' : 'fill-red');
                const percent = Math.round(skill.similarity * 100);
                matrixContainer.innerHTML += `
                    <div class="skill-row">
                        <span style="font-weight: 500;">${skill.skill}</span>
                        <span style="font-size: 0.8rem;">${skill.weight}/10</span>
                        <div class="progress-bg"><div class="progress-fill ${colorClass}" style="width: ${percent}%"></div></div>
                        <span style="text-align: right; font-weight: 600; font-size: 0.85rem;">${percent}%</span>
                    </div>`;
            });

            // Courses (Card Grid)
            const courseContainer = document.getElementById('courseGrid');
            courseContainer.innerHTML = '';
            data.recommendations.slice(0, 4).forEach(course => {
                courseContainer.innerHTML += `
                    <div class="course-card">
                        <div class="course-img"><span style="position:absolute; top:10px; right:10px; background:white; padding:2px 8px; border-radius:4px; font-size:0.7rem; font-weight:700; color:#2563EB;">Score: ${course.relevance_score}</span></div>
                        <div class="course-content">
                            <div>
                                <div class="course-title">${course.name}</div>
                                <div class="course-desc">Fixes: ${course.addresses_gaps.join(', ')}</div>
                            </div>
                            <button onclick="window.open('${course.url}', '_blank')" style="font-size:0.75rem; color:var(--primary); font-weight:600; background:none; border:none; cursor:pointer; padding:0;">View Course →</button>
                        </div>
                    </div>`;
            });

            // AI & Roadmap
            
            let cleanAdvice = data.ai_counselor
                .replace(/\*\*(.*?)\*\*/g, '<strong style="font-weight: bold;">$1</strong>') // Convert ** to bold
                .replace(/\n/g, '<br>'); 

            document.getElementById('aiAdvice').innerHTML = cleanAdvice;
            const roadmapContainer = document.getElementById('roadmapContainer');
            roadmapContainer.innerHTML = '';
            const steps = [{ k: 'month1_2', l: 'Month 1-2' }, { k: 'month3_4', l: 'Month 3-4' }, { k: 'month5_6', l: 'Month 5-6' }];

            steps.forEach(step => {
                const s = data.roadmap[step.k];
                if (s) {
                    roadmapContainer.innerHTML += `
                        <div class="timeline-item">
                            <div class="timeline-dot"></div>
                            <div class="timeline-date">${step.l}</div>
                            <div class="timeline-title">${s.title}</div>
                            <div class="timeline-desc">${stripMarkdown(s.description)}</div>
                        </div>`;
                }
            });
        }

        function updateList(listId, countId, skills, status, icon) {
            const list = document.getElementById(listId);
            const filtered = skills.filter(s => s.status === status);
            document.getElementById(countId).innerText = filtered.length;
            list.innerHTML = '';
            filtered.forEach(s => {
                list.innerHTML += `<li class="gap-item"><i class="fas ${icon}"></i> ${s.skill}</li>`;
            });
        }