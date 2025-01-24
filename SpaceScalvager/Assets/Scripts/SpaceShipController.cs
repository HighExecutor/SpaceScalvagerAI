using System;
using System.Collections;
using System.Collections.Generic;
using System.Numerics;
using TMPro;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.InputSystem;
using Quaternion = UnityEngine.Quaternion;
using Vector2 = UnityEngine.Vector2;
using Vector3 = UnityEngine.Vector3;

public class SpaceShipController : Agent
{
    public float velocity;
    public float angularVelocity;
    private Vector3 movementInput;
    private Vector2 rotateInput;
    private Rigidbody rb;
    private ParticleSystem parts;
    private Transform body;
    public GameObject aimPosition;
    public GameObject[] lasers;
    public TrailRenderer railgunTrail;
    public float shootCooldown;
    private float curShootCooldown;
    public float shootRange;
    private bool canShoot;
    public ParticleSystem shootEffect;
    public CargoUIScript cargoUI;
    private SpaceManager spaceManager;
    private float curMinerals;
    public float maxMinerals;
    public float sellPrice;
    private float credits;
    public bool openHelp;
    public int shipId;

    private Vector3 startPosition;
    private Quaternion startRotate;

    private DecisionRequester decisionRequester;
    private Unity.MLAgents.Policies.BehaviorParameters behaviour;

    private EnvironmentParameters m_ResetParams;
    private TextMeshPro shipIDText;


    // Start is called before the first frame update
    void Start()
    {
        rb = GetComponent<Rigidbody>();
        parts = GetComponentInChildren<ParticleSystem>();
        decisionRequester = GetComponent<DecisionRequester>();
        behaviour = GetComponent<Unity.MLAgents.Policies.BehaviorParameters>();
        spaceManager = GetComponentInParent<SpaceManager>();
        shipIDText = GetComponentInChildren<TextMeshPro>();
        shipIDText.SetText("Ship " + shipId);
        body = transform.GetChild(0);
        curShootCooldown = 0.0f;
        canShoot = true;
        curMinerals = 0.0f;
        credits = 0.0f;
        if (cargoUI != null && shipId == 0)
        {
            cargoUI.SetMaxCargo((int) maxMinerals);
            cargoUI.SetCargo(0.0f);
            int modelMode = 0;
            if (behaviour.BehaviorType == Unity.MLAgents.Policies.BehaviorType.HeuristicOnly)
            {
                modelMode = 1;
            }
            if (behaviour.BehaviorType == Unity.MLAgents.Policies.BehaviorType.InferenceOnly)
            {
                modelMode = 2;
            }
            cargoUI.SetModelControl(modelMode);
            cargoUI.SetHelpEnabled(openHelp);
            cargoUI.SetTimestepsBar(StepCount, MaxStep);
            spaceManager.UpdateStats(true);
        }

        startPosition = transform.position;
        startRotate = transform.rotation;
        
        m_ResetParams = Academy.Instance.EnvironmentParameters;
    }
    
    private void FixedUpdate()
    {
        Move();
        UpdateCooldown();
        if (cargoUI != null && shipId == 0) {
            cargoUI.SetTimestepsBar(StepCount, MaxStep);
        }
    }            
    
    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        float x = actionBuffers.DiscreteActions[0] - 1;
        float y = actionBuffers.DiscreteActions[1] - 1;
        float z = actionBuffers.DiscreteActions[2] - 1;
        float qx = actionBuffers.DiscreteActions[3] - 1;
        float qy = actionBuffers.DiscreteActions[4] - 1;
        bool isShoot = actionBuffers.DiscreteActions[5] == 1;
        // Debug.Log("Acts: " + x + "; " + y + "; " + z + "; " + qx + "; " + qy + "; " + isShoot + "; ");
        if (behaviour.BehaviorType != Unity.MLAgents.Policies.BehaviorType.HeuristicOnly)
        {
            movementInput = new Vector3(x, y, z);
            rotateInput = new Vector2(qx, qy);
        }

        if (isShoot)
        {
            OnShoot();
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var discreteActs = actionsOut.DiscreteActions;
        discreteActs[0] = (int)movementInput.x + 1;
        discreteActs[1] = (int)movementInput.y + 1;
        discreteActs[2] = (int)movementInput.z + 1;
        discreteActs[3] = (int)rotateInput.x + 1;
        discreteActs[4] = (int)rotateInput.y + 1;
        // Debug.Log("Acts: " + discreteActs[0] + "; " + discreteActs[1] + "; " + discreteActs[2] + "; " + discreteActs[3] + "; " + discreteActs[4]);
        if (curShootCooldown == shootCooldown)
        {
            discreteActs[5] = 1;
        }
        else
        {
            discreteActs[5] = 0;
        }
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // sensor.AddObservation(transform.localRotation.eulerAngles / 360f); // 3
        sensor.AddObservation(transform.localRotation); // 4
        sensor.AddObservation(rb.velocity / velocity); // 3
        // sensor.AddObservation(rb.angularVelocity); // 3
        Dictionary<String, object> spaceObservations = spaceManager.GetObservations(shipId);
        sensor.AddObservation((Vector3)spaceObservations["gate"] / velocity); // 3
        List<Vector3> mineralsDists = (List<Vector3>)spaceObservations["mineralsDists"];
        List<Vector3> meteorsDists = (List<Vector3>)spaceObservations["meteorsDists"];
        sensor.AddObservation((float)spaceObservations["meteorsNumber"]); // 1
        for (int i = 0; i < 3; i++)
        {
            sensor.AddObservation(meteorsDists[i] / velocity);  // 3
            Debug.DrawLine(transform.position, transform.position + meteorsDists[i], Color.yellow);
        }
        sensor.AddObservation((float)spaceObservations["mineralsNumber"]); // 1
        for (int i = 0; i < 3; i++)
        {
            sensor.AddObservation(mineralsDists[i] / velocity); // 3
        }
        sensor.AddObservation((Vector3)spaceObservations["shipDist"] / velocity); // 3


        sensor.AddObservation(curShootCooldown); // 1
        sensor.AddObservation(curMinerals / maxMinerals); // 1
        // Total obs: 35
    }

    public override void OnEpisodeBegin()
    {
        Debug.Log("Episode START: shipd " + shipId);
        
        if (shipId == 0)
        {
            int mMaxSteps = (int)m_ResetParams.GetWithDefault("max_steps", MaxStep);
            if (MaxStep != mMaxSteps)
            {
                SetMaxStep(mMaxSteps);
                Debug.Log("Cur max_steps = " + mMaxSteps);
            }
            Time.timeScale = (int)m_ResetParams.GetWithDefault("time_scale", Time.timeScale);
            // Reset all ships first
            spaceManager.ResetTransformAll();
            // Reset meteors and minerals in space
            spaceManager.ResetSpace();
        }        
        credits = 0.0f;
        float mSellPrice = m_ResetParams.GetWithDefault("sell_price", sellPrice);
        if (sellPrice != mSellPrice)
        {
            sellPrice = mSellPrice;
            Debug.Log("Cur sell_price = " + mSellPrice);
        }

        if (cargoUI != null && shipId == 0)
        {
            cargoUI.SetCargo(0.0f);
            cargoUI.SetCredutValue(0.0f);
            cargoUI.SetTimestepsBar(StepCount, MaxStep);
            spaceManager.UpdateStats(true);
        }
    }

    public void ResetTransform()
    {
        transform.position = startPosition;
        transform.rotation = startRotate;
        rb.angularVelocity = Vector3.zero;
        rb.velocity = Vector3.zero;
        movementInput = Vector3.zero;
        rotateInput = Vector2.zero;
        curMinerals = 0.0f;
        if (cargoUI != null && shipId == 0)
        {
            cargoUI.SetCargo(0.0f);
        }
    }

    private void Move()
    {
        if (shipId == 0)
        {
            Debug.Log($"Move: {movementInput}");
        }

        // move
        if (movementInput != Vector3.zero)
        {
            Vector3 forward = transform.forward * movementInput.z;
            Vector3 left = transform.right * movementInput.x;
            Vector3 up = transform.up * movementInput.y;
            Vector3 moveVector = (forward + left + up);
            rb.AddForce(moveVector * velocity * Time.fixedDeltaTime);

        }
        float bodyAngle = body.rotation.eulerAngles.z;
        if (bodyAngle > 180)
        {
            bodyAngle -= 360;
        }
        if (bodyAngle < -180)
        {
            bodyAngle += 360;
        }
        if (movementInput.x != 0)
        {
            bodyAngle -= Mathf.Sign(movementInput.x) * 35 * Time.fixedDeltaTime;
            bodyAngle = Mathf.Clamp(bodyAngle, -15, 15);
            body.rotation = Quaternion.Euler(body.rotation.eulerAngles.x, body.rotation.eulerAngles.y, bodyAngle);
        }
        else if (body.rotation.eulerAngles.z != 0)
        {
            bodyAngle -= Mathf.Sign(bodyAngle) * 35 * Time.fixedDeltaTime;
            bodyAngle = Mathf.Clamp(bodyAngle, -15, 15);
            if (Mathf.Abs(bodyAngle) < 2)
            {
                bodyAngle = 0;
            }
            body.rotation = Quaternion.Euler(body.rotation.eulerAngles.x, body.rotation.eulerAngles.y, bodyAngle);
        }
        if (movementInput.z > 0 && !parts.isPlaying)
        {
            parts.Play();
        }
        else if (parts.isPlaying)
        {
            parts.Pause();
            parts.Clear();
        }

        Vector3 curRotation = transform.rotation.eulerAngles;
        if (curRotation.x > 180)
        {
            curRotation.x -= 360;
        }
        if (curRotation.x < -180)
        {
            curRotation.x += 360;
        }
        if (rotateInput.x != 0 || rotateInput.y != 0 || curRotation.z != 0)
        {
            float horAngle = curRotation.y + rotateInput.x * angularVelocity * Time.fixedDeltaTime;
            float vertAngle = curRotation.x + rotateInput.y * angularVelocity * Time.fixedDeltaTime;
            vertAngle = Mathf.Clamp(vertAngle, -45.0f, 45.0f);
            float diagAngle = curRotation.z;
            if (diagAngle != 0)
            {
                if (diagAngle > 180)
                {
                    diagAngle -= 360;
                }

                if (diagAngle < -180)
                {
                    diagAngle += 360;
                }

                diagAngle -= Mathf.Sign(diagAngle) * angularVelocity * Time.fixedDeltaTime;
                if (Mathf.Abs(diagAngle) < 2)
                {
                    diagAngle = 0;
                }
            }
            transform.rotation = Quaternion.Euler(vertAngle, horAngle, diagAngle);
        }
    }

    void OnMove(InputValue inputValue)
    {
        movementInput = inputValue.Get<Vector3>();
    }

    void OnRotate(InputValue inputValue)
    {
        rotateInput = inputValue.Get<Vector2>();
    }

    void OnShoot()
    {
        // trigger raycast and start trail animation
        if (canShoot)
        {
            Vector3 direction = transform.forward.normalized;
            float range = shootRange;
            LayerMask mask = LayerMask.GetMask("Objects");
            if (Physics.Raycast(aimPosition.transform.position, direction, out RaycastHit hit, shootRange, mask))
            {
                range = Vector3.Distance(aimPosition.transform.position, hit.point);
                ParticleSystem hitEffect = Instantiate(shootEffect, hit.point, Quaternion.identity);
                Destroy(hitEffect.gameObject, 0.5f);
                if (hit.collider.CompareTag("Meteor"))
                {
                    MeteorScript meteor = hit.collider.GetComponent<MeteorScript>();
                    meteor.TakeHit();
                    AddReward(0.1f);
                }
            }
            for (int l = 0; l < lasers.Length; l++)
            {
                TrailRenderer trail = Instantiate(railgunTrail, lasers[l].transform);
                StartCoroutine(SpawnTrail(trail, range, shootRange));

            }
            canShoot = false;
            curShootCooldown = shootCooldown;
        }
    }
    private void UpdateCooldown()
    {
        if (curShootCooldown > 0)
        {
            curShootCooldown -= Time.fixedDeltaTime;
            if (curShootCooldown <= 0)
            {
                canShoot = true;
                curShootCooldown = 0.0f;
            }
        }
    }

    void OnHelp()
    {
        if (shipId == 0)
        {
            openHelp = !openHelp;
            if (cargoUI != null)
            {
                cargoUI.SetHelpEnabled(openHelp);
            }
        }
    }

    void OnHeuristic()
    {
        if (shipId == 0)
        {
            if (behaviour.BehaviorType != Unity.MLAgents.Policies.BehaviorType.Default)
            {
                if (behaviour.BehaviorType == Unity.MLAgents.Policies.BehaviorType.HeuristicOnly)
                {
                    if (behaviour.Model != null)
                    {
                        decisionRequester.DecisionPeriod = 4;
                        behaviour.BehaviorType = Unity.MLAgents.Policies.BehaviorType.InferenceOnly;
                        cargoUI.SetModelControl(2);
                    }
                }
                else
                {
                    decisionRequester.DecisionPeriod = 1;
                    behaviour.BehaviorType = Unity.MLAgents.Policies.BehaviorType.HeuristicOnly;
                    cargoUI.SetModelControl(1);
                }
            }
            else
            {
                // cargoUI.SetModelControl(0);
                // Start with default but if change then go to heuristics
                decisionRequester.DecisionPeriod = 1;
                behaviour.BehaviorType = Unity.MLAgents.Policies.BehaviorType.HeuristicOnly;
                cargoUI.SetModelControl(1);
            }
        }
    }

    void OnReset()
    {
        spaceManager.EndEpisodeAll();
    }

    IEnumerator SpawnTrail(TrailRenderer trail, float range, float maxRange)
    {
        float time = 0f;
        Vector3 startPosition = trail.transform.position;
        Vector3 endPoint = startPosition + trail.transform.up.normalized * range;
        float rangeRate = range / maxRange;
        while (time < rangeRate)
        {
            trail.transform.position = Vector3.Lerp(startPosition, endPoint, time / rangeRate);
            time += Time.fixedDeltaTime / trail.time;
            yield return null;
        }
        trail.transform.position = endPoint;
        Destroy(trail.gameObject, trail.time);

    }

    public void TakeMineral(float amount)
    {
        curMinerals = Mathf.Min(maxMinerals, curMinerals + amount);
        if (cargoUI != null && shipId == 0)
        {
            cargoUI.SetCargo(curMinerals);
        } else
        {
            spaceManager.UpdateStats(false);
        }

        AddReward(0.1f);
        
    }
    
    public void SellMinerals()
    {
        AddReward(curMinerals / maxMinerals * sellPrice);
        credits += curMinerals * sellPrice;
        curMinerals = 0.0f;
        if (cargoUI != null && shipId == 0)
        {
            cargoUI.SetCargo(0.0f);
            cargoUI.SetCredutValue(credits);
        } else
        {
            spaceManager.UpdateStats(false);
        }
    }

    public void AddCustomReward(float reward)
    {
        AddReward(reward);
    }
    
    public void SetMaxStep(int maxSteps)
    {
        MaxStep = maxSteps;
    }

    public int GetCurCargo()
    {
        return (int)curMinerals;
    }

    public int GetCurCredits()
    {
        return (int)credits;
    }    
}